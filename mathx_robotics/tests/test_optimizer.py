import mathx.learning.models as clmodels
# from mathx.learning.q_function import QFunctionComposite
# from mathx.learning.policy_function import PolicyFunction
from mathx.learning.database import Database
# from mathx.learning.optimizer import Optimizer
# from mathx.learning.hyperparameters import Hyperparameters
from mathx.learning.custom_agent import CustomAgent
from mathx.learning.sb3_agent import SB3Agent
from quadratic_system import QuadraticSystem
import numpy as np
from mathx.learning import log
from test_config import *

# import matplotlib
# matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
import time
import dm_env

def gui_loop(interval):
  manager = plt._pylab_helpers.Gcf.get_active()
  if manager is not None:
    canvas = manager.canvas
    if canvas.figure.stale:
      canvas.draw_idle()
    #plt.show(block=False)
    canvas.start_event_loop(interval)
  else:
    time.sleep(interval)

def test_optimizer():
  log.initialize("optimizer.log.txt")

  rand=np.random.default_rng(56425)
  system=QuadraticSystem(2,False)

  batch_size=4
  agent=CustomAgent({"beta":.9,"batch_size":batch_size},system.state_space(),system.action_space())
  # agent=SB3Agent({"beta":.9,"batch_size":batch_size},system.state_space(),system.action_space())

  database=Database()

  num_episodes=100
  num_steps=20

  for episode_idx in range(num_episodes):
    log.info(f"episode {episode_idx}")
    system.reset(rand)
    for step_idx in range(num_steps):
      state0=system.state()
      action0=system.action_space().sample(rand)
      system.transition(action0)
      feedback=system.feedback()
      state1=system.state()
      discount=np.array(1)#0 if step_idx==(num_steps-1) else 1)
      point=(state0,action0,state1,feedback,discount)
      database.add(point)

      if step_idx==0:
        agent.observe_first(dm_env.TimeStep(step_type=dm_env.StepType.FIRST,reward=None,discount=None,observation=state0))
      agent.observe(action0,dm_env.TimeStep(step_type=dm_env.StepType.MID,reward=feedback,discount=discount,observation=state1))

      log.info(f"s={state1[:2]} a={action0} f={feedback[0]}")

  database.print_stats()

  # num_iterations=data.size()*100
  num_iterations=10**9
  # num_iterations=2

  # model_params={
  #   **clmodels.architecture_params,
  #   "state_space":system.state_space(),
  #   "action_space":system.action_space()
  # }

  # q=QFunctionComposite(model_params)
  # policy=PolicyFunction(model_params)

  # optimizer=Optimizer(Hyperparameters.get_default(),
  #                     q,policy)

  if debug_visualize():
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 6), squeeze=False)
    plt.ion()
    plt.show()

  for iteration_idx in range(num_iterations):
    # batch=database.sample(batch_size,rand)
    # q_obj,policy_obj=optimizer.update(batch)
    # log.info(batch)
    q_obj,policy_obj=agent.update(wait=True)

    log.info(f"iteration={iteration_idx} q_objective={q_obj: 16g} policy_objective={policy_obj: 16g}")
    # log.info(f"iteration={iteration_idx}")

    if debug_visualize():
      if iteration_idx%1000==0:
        data=database.get()

        x=[float(d[2][0]) for d in data]
        y=[float(d[2][1]) for d in data]

        v=[0]*database.size()
        px=[0]*database.size()
        py=[0]*database.size()
        for i,pt in enumerate(data):
          p=agent.select_action(pt[0])
          px[i]=p[0]
          py[i]=p[1]
          v[i]=float(agent.q(pt[0],p))

        # Feedback plot
        axs[0,0].clear()
        axs[0,0].title.set_text("feedback")
        feedback_plot=axs[0,0].scatter(
          x=x,y=y,
          c=[float(d[3][0]) for d in data])
        # axs[1,0].colorbar()
        # fig.colorbar(feedback_plot,ax=axs[1,0])

        # Action plot
        axs[1,0].clear()
        axs[1,0].title.set_text("action0")
        action_plot=axs[1,0].quiver(
          x,y,
          [float(d[1][0]) for d in data],
          [float(d[1][1]) for d in data])

        # Policy plot
        axs[2,0].clear()
        axs[2,0].title.set_text("policy")
        policy_plot=axs[2,0].quiver(x,y,px,py)

        # Value plot
        axs[3,0].clear()
        axs[3,0].title.set_text("value")
        v_plot=axs[3,0].scatter(
          x=x,y=y,c=v)
        # axs[0,0].colorbar()
        # fig.colorbar(q_plot,ax=axs[0,0])

        plt.draw()

      if iteration_idx%100==0:
        # plt.pause(.001)
        gui_loop(.001)

if __name__=="__main__":
  test_optimizer()
