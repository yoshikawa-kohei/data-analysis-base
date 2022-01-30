python runner.py TrainWorkflow --local-scheduler

if [ "$1" = "test" ]; then
    python runner.py TestWorkflow --local-scheduler
fi