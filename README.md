# Spine_Seg

**声明：该代码仅用于[**第二届** **CSIG** **图像图形技术挑战赛**](https://www.spinesegmentation-challenge.com/)，引用请注明来源。**

**作者：SHIJUN （shijun18@mail.ustc.edu.cn）**

## 代码结构

- Spine_Seg
  - converter  (数据处理和分析模块)
    - meta_data
    - static_files
    - nii_reader.py  (nii解析)
    - ......
  - data_utils (数据加载模块)
    - data_loader.py
    - transformer.py
  - loss (损失函数定义)
  - metrics.py (指标定义)
  - config.py (参数文件)
  - trainer.py (训练主类)
  - run.py (运行脚本)
  - test.py (测试脚本)
  - ensemble.py (结果融合)
  - prepare_submission.py (准备提交结果)
  - utils.py (工具箱)

## 代码执行

### 运行环境

依赖包: 见 `requirements.txt`，ps:可能不全，遗漏自补。

硬件:  `NVIDIA A100` *1

### 实验设置

| Net           | Encoder  | lr   | shape   | Batch size |
| ------------- | -------- | ---- | ------- | ---------- |
| Deeplabv3plus | Resnet50 | 1e-3 | 512x512 | 16         |

| lr_scheduler | optimizer | loss                      | Fold num |
| ------------ | --------- | ------------------------- | -------- |
| MultiStepLR  | Adam      | Topk Cross Entropy (k=20) | 5        |

### 数据生成

- 将训练数据路径写入 `converter/static_files/spine.json`中的 `nii_path`，并指定存储路径 `npy_data`和 `2d_data/save_path`，前者为3d格式，后者为2d切片数据，`spine.json`文件如下。

  ```json
  {
      "nii_path":"../dataset/train",
      "npy_path":"../dataset/npy_data",
      "metadata_path":"./meta_data/spine_metadata.csv",
      "annotation_num":19,
      "annotation_list":[
          "S",
          "L5",
          "L4",
          "L3",
          "L2",
          "L1",
          "T12",
          "T11",
          "T10",
          "T9",
          "L5/S",
          "L4/L5",
          "L3/L4",
          "L2/L3",
          "L1/L2",
          "T12/L1",
          "T11/T12",
          "T10/T11",
          "T9/T10"
      ],
      "2d_data":{
          "save_path":"../dataset/2d_data",
          "crop":0,
          "shape":[512,512],
          "csv_path":"./static_files/spine.csv"
      },
      "mean_std":{
          "mean":239,
          "std":257
      }
  }
  ```
- nii 转 HDF5

  ```shell
  cd converter
  python nii2npy.py
  ```
- 生成2d切片

  ```shell
  python prepare_data.py
  ```

### 训练过程

将所有目标分为两类，椎骨（10种-Part_10）与椎间盘（9种-Part_9）分开训练。

- Part_10

  - 修改 `config.py`

    ```python
    ROI_NUMBER = [1,2,3,4,5,6,7,8,9,10]
    ```
  - 运行 `run.py`

    ```shell
    python run.py -m train-cross
    ```
- Part_9

  - 修改 `config.py`

    ```python
    ROI_NUMBER = [11,12,13,14,15,16,17,18,19]
    ```
  - 运行 `run.py`

    ```shell
    python run.py -m train-cross
    ```

### 测试过程

- 生成结果

  ```shell
  python test.py -tp ./dataset/test2/MR -cp ../trained_model/
  ```
- 结果融合

  ```shell
  python ensemble.py
  ```
- 准备提交结果及后处理

  ```shell
  python prepare_submission.py
  ```
- 打包文件

  将 `segmentation_results` 压缩成 `.zip`

  ```shell
  cd result/v4.3-all
  zip -r -q shijun.zip segmentation_results
  ```
