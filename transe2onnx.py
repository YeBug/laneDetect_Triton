import torch
import torch.onnx 

class Transe2onnx:
    def transe2onnx(self, model):
        model = model.to(torch.device('cuda'))
        model.eval()
        batch_size = 1 								# 批处理大小
        input_shape = (3, 360, 640) 				# 输入数据  

        x = torch.randn(batch_size,*input_shape).cuda() 	# 生成张量
        export_onnx_file = "test.onnx" 				# 目的ONNX文件名
        torch.onnx.export(model,
                    x,
                    export_onnx_file,   
                    input_names=['input'],          # 输入别名
                    output_names=['predict_lanes'], # 输出别名
                    opset_version=11,
                    verbose=False)

if __name__ == "__main__":
    transe = Transe2onnx()
    transe.transe2onnx()
