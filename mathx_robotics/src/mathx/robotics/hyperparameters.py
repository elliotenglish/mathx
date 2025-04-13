from dataclasses import dataclass, asdict
import os
import json

@dataclass(frozen=True)
class Hyperparameters:
  q_learning_rate: float
  policy_learning_rate: float
  momentum: float
  weight_decay: float
  beta: float
  bellman_error: float
  use_q_target_gradient: bool
  policy_optimality: float
  cql: float
  out_of_distribution_regularization: float
  action_regularization: float
  feedback_overestimate_penalty: float
  target_learning_rate: float

  @staticmethod
  def get_default(**kwargs):
    with open(os.path.join(os.path.dirname(__file__),"../../../data/hyperparameters.json"),"r") as f:
      data=json.load(f)
    return Hyperparameters(**{**data,**kwargs})
