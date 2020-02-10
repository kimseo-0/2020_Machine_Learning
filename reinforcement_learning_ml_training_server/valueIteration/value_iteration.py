from reinforcement_learning_ml_training_server.valueIteration.environment import GraphicDisplay, Env


class ValueIteration:
    def __init__(self, env):
        self.env = env
        # env.get_all_states(): 모든 상태를 담은 배열을 반환한다
        # env.possible_actions = [0, 1, 2, 3]: 상, 하, 좌, 우를 나타낸다
        # env.get_reward(state, action): 어떤 상태에서, 어떤 액션에 주어지면, 리워드를 반환한다
        # env.state_after_action(state, action_index): 어떤 상태에서, 어떤 액션의 인덱스가 주어지면, 다음 상태를 반환한다
        self.value_table = [[0.00] * env.width for _ in range(env.height)]
        self.discount_factor = 0.9

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)

    # 아래의 함수를 완성하세요
    def value_update_by_policy(self):
        update_value_table = self.value_table
        for state in env.get_all_states():
            current_action_value_list = []
            for action in env.possible_actions:
                next_state = env.state_after_action(state, action)
                reward = env.get_reward(state, action)
                next_value = self.get_value(next_state)
                current_action_value = reward + self.discount_factor * next_value
                current_action_value_list.append(current_action_value)
            print(current_action_value_list)
            update_value_table[state[0]][state[1]] = round(max(current_action_value_list), 2)

        self.value_table = update_value_table

    def get_action(self, state):
        if state == [2, 2]:
            return []
        chosen_max_value = -99999
        list_of_chosen_actions = []
        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            temp_value = reward + self.discount_factor * next_value
            if temp_value > chosen_max_value:
                list_of_chosen_actions.clear()
                list_of_chosen_actions.append(action)
                chosen_max_value = temp_value
            elif temp_value == chosen_max_value:
                list_of_chosen_actions.append(action)
        return list_of_chosen_actions


if __name__ == "__main__":
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()
