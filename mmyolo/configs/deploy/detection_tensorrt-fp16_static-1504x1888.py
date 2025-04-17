_base_ = ['./base_static.py']
onnx_config = dict(input_shape=(1888, 1504))
backend_config = dict(
    type='tensorrt',
    common_config=dict(fp16_mode=True, max_workspace_size=1 << 35),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 1504, 1888],
                    opt_shape=[1, 3, 1504, 1888],
                    max_shape=[1, 3, 1504, 1888])))
    ])
use_efficientnms = False  # whether to replace TRTBatchedNMS plugin with EfficientNMS plugin # noqa E501
