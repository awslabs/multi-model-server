|CODE|Test Types                                                                          |Comments                                                                                                                                   |
|----|------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
|A   |Operation time is deterministic for given inputs and env configuration              |Model Inference Time = (Wait time for worker to be free) + (Time taken by worker to infer)                                                 |
|B   |Minimum drift in user sla and system metrics over time                              |Current architecture doesnt support recycling of workers. Need to check whether params specified in sheet 2 remain within acceptable bounds|
|C   |Demonstrate performance isolation across registered models                          |User uploaded handlers cannot cause denial of service (dos) for other model type inference                                                 |
|D   |Operations should scale linearly on node                                            |Time to scale N workers == ( Time to scale 1 worker ) * N                                                                                  |
|E   |Demonstrate expected service concurrency commensurate with environment configuration|Setup environment in a way to minimize false positives/negatives                                                                           |
|F   |Demonstrate preservation of performance characteristics                             |Do not spawn additional workers, accept model registrations if they hamper system SLAs. Is this implemented currently?                     |
|G   |Demonstrate cleanup releases system resources                                       |Unregistering model should free up commensurate resources held by workers                                                                  |
|H   |Demonstrate that cleanup/termination of operations should be graceful               |Scale down should wait for current inference operation to succeed. Is this current behavior?                                               |
|I   |Demonstrate that operations rollback in case request cannot be satisifed            |Ongoing inference should complete before scaledown operation is allowed to start                                                           |
|J   |Demonstrate that operations are idempotent                                          |Multiple simultaneous scale operations with the same parameter value should result in the same system state                                |

|API|CODE|YAML|
|---|----|---|
|Register Model|A,B,F,J|[register_unregister.yaml](tests/register_unregister.yaml)|
|Inference|A,B,C|[inference_single_worker.yaml](tests/inference_single_worker.yaml), [inference_multiple_worker.yaml](tests/inference_multiple_worker.yaml)|
|Batch Inference|A,B,C||
|Custom Model Handlers|C||
|Scale Workers - UP/DOWN|D,G,I,F,J|[scale_up_workers.yaml](tests/scale_up_workers.yaml), [scale_down_workers.yaml](tests/scale_down_workers.yaml)|
|Unregister Models|D,G,I,J|[register_unregister.yaml](tests/register_unregister.yaml)|
|Health Check|A,B,E||
|API Description|A,B,E||
|Model Describe|A,B,E|[model_description.yaml](tests/model_description.yaml)|
|List Models|A,B,E|[list_models.yaml](tests/list_models.yaml)|
