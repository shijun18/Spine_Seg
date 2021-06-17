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
  - eval.py (验证脚本)
  - ensemble.py (结果融合)
  - prepare_submission.py (准备提交结果)
  - utils.py (工具箱)

## 代码执行

### 运行环境

依赖包见 `requirements.txt`，ps:可能不全，遗漏自补。

硬件NVIDIA A100 *2

### 实验设置

| Net           | Encoder  | lr   | shape   | Batch size |
| ------------- | -------- | ---- | ------- | ---------- |
| Deeplabv3plus | Resnet50 | 1e-3 | 512x512 | 16         |

| lr_scheduler | optimizer | loss                      | Fold num |
| ------------ | --------- | ------------------------- | -------- |
| MultiStepLR  | Adam      | Topk Cross Entropy (k=20) | 5        |

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
  python test.py
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
  zip -r -q shijun.zip segmentation_results
  ```
