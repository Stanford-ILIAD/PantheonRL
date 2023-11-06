pantheonrl.algos.modular.policies.ModularPolicy
===============================================

.. currentmodule:: pantheonrl.algos.modular.policies

.. autoclass:: ModularPolicy
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   
      ~ModularPolicy.add_module
      ~ModularPolicy.apply
      ~ModularPolicy.bfloat16
      ~ModularPolicy.buffers
      ~ModularPolicy.build_mlp_action_value_net
      ~ModularPolicy.children
      ~ModularPolicy.cpu
      ~ModularPolicy.cuda
      ~ModularPolicy.do_init_weights
      ~ModularPolicy.double
      ~ModularPolicy.eval
      ~ModularPolicy.evaluate_actions
      ~ModularPolicy.extra_repr
      ~ModularPolicy.extract_features
      ~ModularPolicy.float
      ~ModularPolicy.forward
      ~ModularPolicy.get_action_logits_from_obs
      ~ModularPolicy.get_buffer
      ~ModularPolicy.get_extra_state
      ~ModularPolicy.get_parameter
      ~ModularPolicy.get_submodule
      ~ModularPolicy.half
      ~ModularPolicy.init_weights
      ~ModularPolicy.ipu
      ~ModularPolicy.is_vectorized_observation
      ~ModularPolicy.load
      ~ModularPolicy.load_from_vector
      ~ModularPolicy.load_state_dict
      ~ModularPolicy.make_action_dist_net
      ~ModularPolicy.make_features_extractor
      ~ModularPolicy.modules
      ~ModularPolicy.named_buffers
      ~ModularPolicy.named_children
      ~ModularPolicy.named_modules
      ~ModularPolicy.named_parameters
      ~ModularPolicy.obs_to_tensor
      ~ModularPolicy.overwrite_main
      ~ModularPolicy.parameters
      ~ModularPolicy.parameters_to_vector
      ~ModularPolicy.predict
      ~ModularPolicy.register_backward_hook
      ~ModularPolicy.register_buffer
      ~ModularPolicy.register_forward_hook
      ~ModularPolicy.register_forward_pre_hook
      ~ModularPolicy.register_full_backward_hook
      ~ModularPolicy.register_full_backward_pre_hook
      ~ModularPolicy.register_load_state_dict_post_hook
      ~ModularPolicy.register_module
      ~ModularPolicy.register_parameter
      ~ModularPolicy.register_state_dict_pre_hook
      ~ModularPolicy.requires_grad_
      ~ModularPolicy.save
      ~ModularPolicy.scale_action
      ~ModularPolicy.set_extra_state
      ~ModularPolicy.set_freeze_main
      ~ModularPolicy.set_freeze_module
      ~ModularPolicy.set_freeze_partner
      ~ModularPolicy.set_training_mode
      ~ModularPolicy.share_memory
      ~ModularPolicy.state_dict
      ~ModularPolicy.to
      ~ModularPolicy.to_empty
      ~ModularPolicy.train
      ~ModularPolicy.type
      ~ModularPolicy.unscale_action
      ~ModularPolicy.xpu
      ~ModularPolicy.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~ModularPolicy.T_destination
      ~ModularPolicy.call_super_init
      ~ModularPolicy.device
      ~ModularPolicy.dump_patches
      ~ModularPolicy.squash_output
      ~ModularPolicy.features_extractor
      ~ModularPolicy.optimizer
      ~ModularPolicy.training
   
   