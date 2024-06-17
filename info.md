For depth estimation, several state-of-the-art models are highly recommended based on their performance and widespread use:

1. **DPT-Large**: This model is one of the top-performing models for monocular depth estimation. It is trained on a large dataset and offers excellent accuracy. It is available on Hugging Face under the model name "Intel/dpt-large"【148†source】【149†source】.

2. **DPT-Hybrid MiDaS**: This model, also available on Hugging Face, provides a good balance between performance and model size. It is well-suited for environments where computational resources might be more limited compared to what DPT-Large requires. You can find it under the model name "Intel/dpt-hybrid-midas"【148†source】.

3. **GLPN fine-tuned on NYU**: The Global-Local Path Networks model fine-tuned on the NYU Depth V2 dataset uses the SegFormer framework and is also noted for its performance in depth estimation tasks. It is available under the model name "vinvino02/glpn-nyu"【148†source】.

Among these, the **DPT-Large** model generally provides the best performance but requires significant computational resources. If you need a balance between performance and resource usage, the **DPT-Hybrid MiDaS** model is a solid choice. 

You can implement and test these models using the Hugging Face Transformers library, which simplifies their deployment and usage.

Here are links to the respective models on Hugging Face:
- [DPT-Large](https://huggingface.co/Intel/dpt-large)
- [DPT-Hybrid MiDaS](https://huggingface.co/Intel/dpt-hybrid-midas)
- [GLPN fine-tuned on NYU](https://huggingface.co/vinvino02/glpn-nyu)

These resources should help you integrate state-of-the-art depth estimation into your projects effectively.



MiDaS 3.1
- For highest quality: [dpt_beit_large_512](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt)
- For moderately less quality, but better speed-performance trade-off: [dpt_swin2_large_384](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt)
- For embedded devices: [dpt_swin2_tiny_256](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt), [dpt_levit_224](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt)
- For inference on Intel CPUs, OpenVINO may be used for the small legacy model: openvino_midas_v21_small [.xml](https://github.com/isl-org/MiDaS/releases/download/v3_1/openvino_midas_v21_small_256.xml), [.bin](https://github.com/isl-org/MiDaS/releases/download/v3_1/openvino_midas_v21_small_256.bin)

MiDaS 3.0: Legacy transformer models [dpt_large_384](https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt) and [dpt_hybrid_384](https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt)

MiDaS 2.1: Legacy convolutional models [midas_v21_384](https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt) and [midas_v21_small_256](https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt) 