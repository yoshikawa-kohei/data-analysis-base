import luigi

from pipelines.workflow import TrainWorkflow

luigi.run(["TrainWorkflow", "--workers", "1", "--local-scheduler"])
