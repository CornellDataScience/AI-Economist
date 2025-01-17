import numpy as np
from ai_economist.foundation.base.base_component import BaseComponent, component_registry

@component_registry.add
class GetEducated(BaseComponent):
    """
    Environments expand the agents' state/action spaces by querying:
        get_n_actions
        get_additional_state_fields
    Environments expand their dynamics by querying:
        component_step
        generate_observations
        generate_masks
    Environments expand logging behavior by querying:
        get_metrics
        get_dense_log
    Because they are built as Python objects, component instances can also be
    stateful. Stateful attributes are reset via calls to:
        additional_reset_steps
    """
    name = "GetEducated"
    required_entities = ["Coin", "Labor", "build_skill"]  
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        tuition=100, # same tuition cost as building 10 houses <- tweak later
        education_labor=100.0,
        skill_gain = 1
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)
        self.tuition = int(tuition)
        self.skill_gain = float(skill_gain)
        assert self.tuition >= 0
        self.education_labor = float(education_labor)
        assert self.education_labor >= 0
        # self.skill = int(skill)
        # self.number_times_educated = 0
        self.educates = []

    def agent_can_get_educated(self, agent):
        """Return True if agent can actually get educated."""
        # See if the agent has the resources necessary to complete the action

        if agent.state["inventory"]["Coin"] < self.tuition:
            return False

        # # Do nothing if skill is already max
        # if True: # TODO see how to get skill
        #     return False

        # If we made it here, the agent can go to college.
        return True

    def get_additional_state_fields(self, agent_cls_name):
        """
        Return a dictionary of {state_field: reset_val} managed by this Component
        class for agents of type agent_cls_name. This also partially controls reset
        behavior.
        Args:
            agent_cls_name (str): name of the Agent class for which additional states
                are being queried. For example, "BasicMobileAgent".
        Returns:
            extra_state_dict (dict): A dictionary of {"state_field": reset_val} for
                each extra state field that this component adds/manages to agents of
                type agent_cls_name. This extra_state_dict is incorporated into
                agent.state for each agent of this type. Note that the keyed fields
                will be reset to reset_val when the environment is reset.
        """
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicMobileAgent":
            return {"tuition_payment": float(self.tuition)} # check
        raise NotImplementedError

    def additional_reset_steps(self):
        # reset skill level
        world = self.world
        for agent in world.agents:
            if self.skill_dist == "none":
                    sampled_skill = 1
                    pay_rate = 1
            elif self.skill_dist == "pareto":
                sampled_skill = np.random.pareto(4)
                pay_rate = np.minimum(PMSM, (PMSM - 1) * sampled_skill + 1)
            elif self.skill_dist == "lognormal":
                sampled_skill = np.random.lognormal(-1, 0.5)
                pay_rate = np.minimum(PMSM, (PMSM - 1) * sampled_skill + 1)
            else:
                raise NotImplementedError

            agent.state["build_skill"] = float(sampled_skill)

            self.sampled_skills[agent.idx] = sampled_skill

    def get_n_actions(self, agent_cls_name):
        """
        Args:
            agent_cls_name (str): name of the Agent class for which number of actions
                is being queried. For example, "BasicMobileAgent".
        Returns:
            action_space (None, int, or list): If the component does not add any
                actions for agents of type agent_cls_name, return None. If it adds a
                single action space, return an integer specifying the number of
                actions in the action space. If it adds multiple action spaces,
                return a list of tuples ("action_set_name", num_actions_in_set).
                See below for further detail.
        """
        if agent_cls_name == "BasicMobileAgent":
            return 1
        return None

    def generate_masks(self, completions=0):
        masks = {}
        for agent in self.world.agents:
            masks[agent.idx] = np.array([
                agent.state["inventory"]["Coin"] >= self.widget_price and self.available_widget_units > 0
            ])

        return masks

    def component_step(self):
        """
        See base_component.py for detailed description.
        Convert coin to skill for agents that choose to go to school and can.
        """
        
        world = self.world
        build = []
        # Apply any go_to_school actions taken by the mobile agents
        for agent in world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            # This component doesn't apply to this agent!
            if action is None:
                continue

            # NO-OP!
            if action == 0:
                pass

            # Learn! (If you can.)
            elif action == 1:
                if self.agent_can_get_educated(agent):
                    # Remove the resources
                    agent.state["inventory"]["Coin"] -= self.tuition

                    # Receive skills for going to school
                    agent.state["build_skill"] += self.skill_gain
                    # self.payment_max_skill_multiplier += self.skill_gain

                    # Incur the labor cost for going to school
                    agent.state["endogenous"]["Labor"] += self.education_labor

                    # self.number_times_educated += 1

            else:
                raise ValueError

    #   self.builds.append(build)

    def generate_observations(self):
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "skill_gain": self.skill_gain,
                "tuition": self.tuition,
                "build_skill": self.sampled_skills[agent.idx]
            }

        return obs_dict