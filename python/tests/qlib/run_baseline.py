import qlib
from qlib.workflow import R
from qlib.utils import init_instance_by_config
import yaml


def run_workflow():
    # 1. 加载配置
    with open("baseline.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. 初始化 Qlib
    qlib.init(**config["qlib_init"])

    # 3. 初始化模型和数据集
    # 注意：init_instance_by_config 会触发多进程计算
    model = init_instance_by_config(config["task"]["model"])
    dataset = init_instance_by_config(config["task"]["dataset"])

    # 4. 开始实验
    with R.start(experiment_name="workflow_lite"):
        # 训练模型
        print("开始训练模型...")
        model.fit(dataset)

        # 预测预测
        print("开始预测...")
        prediction = model.predict(dataset)
        print("预测结果预览:")
        print(prediction.head())

        print("预测分数的统计信息:")
        print(prediction.describe())

        # 保存预测结果
        R.save_objects(predict=prediction)


# !!! 核心修复在这里 !!!
if __name__ == '__main__':
    run_workflow()