def epsilon_greedy_action(x,epsilon,action_space,rand,policy):
  if(rand.uniform(0,1)>=epsilon):
    action=policy.optimal_action(x)
  else:
    #action=np.zeros(policy.action_space)
    #action[rand.randint(0,policy.action_space-1)]=1
    action=action_space.to_real(action_space.sample(rand))
  return action
