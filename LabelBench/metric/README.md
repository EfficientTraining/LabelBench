# Metrics
To implement a new metric in `metric_impl`, simply inherit the `LabelBench.skeleton.metric_skeleton.Metric` class
and start with the following structure.
```
class YourMetric(Metric):
    metric_name = "Your Metric Name"
    
    def compute(...):
        ...  # Your implementation goes here.
```
`metric_name` should be specified as the name of the new metric. It will also be used to specify the "--metric"
argument when launching experiments.
