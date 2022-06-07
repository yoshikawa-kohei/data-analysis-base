import gokart
import luigi

from pipelines.workflow import TrainWorkflow

if __name__ == "__main__":
    luigi.configuration.LuigiConfigParser.add_config_path("./conf/param.ini")
    gokart.run()
