mkdir -p ./trainlogs

#### MT10 ####

### use fairgrad to solve the optimization ###
nohup python -u main.py setup=metaworld env=metaworld-mt10 agent=fairgrad_state_sac experiment.num_eval_episodes=1 experiment.num_train_steps=2000000 setup.seed=0 replay_buffer.batch_size=1280 agent.multitask.num_envs=10 agent.multitask.should_use_disentangled_alpha=False agent.multitask.should_use_task_encoder=False agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False agent.encoder.type_to_select=identity agent.builder.agent_cfg.fairgrad_alpha=2.0 > trainlogs/mt10_fairgrad-alpha2.0_sd0.log 2>&1 &


#### MT50 ####

### use fairgrad to solve the optimization ###
# nohup python -u main.py setup=metaworld env=metaworld-mt50 agent=fairgrad_state_sac experiment.num_eval_episodes=1 experiment.num_train_steps=2000000 setup.seed=0 replay_buffer.batch_size=1280 agent.multitask.num_envs=50 agent.multitask.should_use_disentangled_alpha=False agent.multitask.should_use_task_encoder=False agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False agent.encoder.type_to_select=identity agent.builder.agent_cfg.fairgrad_alpha=2.0 > trainlogs/mt50_fairgrad-alpha2.0_sd0.log 2>&1 &
