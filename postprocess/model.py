# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import numpy as np
import torch
import torch.nn as nn
from nms import nms
from scipy.interpolate import InterpolatedUnivariateSpline
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output_tensor = pb_utils.get_output_config_by_name(
            model_config, "lanes")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output_tensor['data_type'])


    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype
        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            prediction = pb_utils.get_input_tensor_by_name(request, "prediction").as_numpy()
            nms_output = self.__nms(prediction)
            # Get INPUT0
            decode_output = self.__decode(nms_output)
            lanes = self.__pred2lanes(decode_output)
            lanes = np.array(lanes)
            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("lanes",
                                           lanes.astype(output0_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def __nms(self, batch_proposals,  nms_thres=45, nms_topk=8, conf_threshold=0.2):
        softmax = nn.Softmax(dim=1)
        batch_proposals = torch.Tensor(batch_proposals).cuda()
        proposals_list = []
        for proposals in batch_proposals:
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            scores = softmax(proposals[:, :2])[:, 1]
            above_threshold = scores > conf_threshold
            proposals = proposals[above_threshold]
            scores = scores[above_threshold]
            if proposals.shape[0] == 0:
                proposals_list.append(proposals[[]])
                continue
            keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
            keep = keep[:num_to_keep]
            proposals = proposals[keep]
            proposals_list.append(proposals)

        return proposals_list

    def __decode(self, proposals_list):
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals in proposals_list:
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            pred = self.__proposals_to_pred(proposals)
            decoded.append(pred)

        return decoded

    def __proposals_to_pred(self, proposals):
        anchor_ys = torch.linspace(1, 0, steps=72, dtype=torch.float32)
        anchor_ys = anchor_ys.to(proposals.device)
        anchor_ys = anchor_ys.double()
        lanes = []
        for lane in proposals:
            lane_xs = lane[5:] / 640
            start = int(round(lane[2].item() * (72 - 1)))
            length = int(round(lane[4].item()))
            end = start + length - 1
            end = min(end, len(anchor_ys) - 1)
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def __pred2lanes(self, preds):
        y_samples =  [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330,
         340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550,
          560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
        all_lanes = []
        for pred in preds:
            ys = np.array(y_samples) / 720
            lanes = []
            for lane in pred:
                xs = lane(ys)
                invalid_mask = xs < 0
                lane = (xs * 1280).astype(int)
                lane[invalid_mask] = -2
                lanes.append(lane.tolist())
            all_lanes.append(lanes)
        return all_lanes

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')


class Lane:
    def __init__(self, points=None, invalid_value=-2., metadata=None):
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(points[:, 1], points[:, 0], k=min(3, len(points) - 1))
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01

        self.metadata = metadata or {}

    def __repr__(self):
        return '[Lane]\n' + str(self.points) + '\n[/Lane]'

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)

        lane_xs[(lane_ys < self.min_y) | (lane_ys > self.max_y)] = self.invalid_value
        return lane_xs

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter < len(self.points):
            self.curr_iter += 1
            return self.points[self.curr_iter - 1]
        self.curr_iter = 0
        raise StopIteration

