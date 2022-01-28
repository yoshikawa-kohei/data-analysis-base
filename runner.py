import luigi

from pipelines.workflow import MainWorkflow

luigi.run(["MainWorkflow", "--workers", "1", "--local-scheduler"])
