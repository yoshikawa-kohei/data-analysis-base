import luigi
import gokart
from pipelines.workflow import TrainWorkflow, TestWorkflow


if __name__ == "__main__":
    luigi.configuration.LuigiConfigParser.add_config_path("./conf/param.ini")
    gokart.run()
