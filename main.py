# Usage: streamlit run main.py

import streamlit as st
import numpy as np

from connectfour import ConnectFour
from mcts import MCTS
import matplotlib.pyplot as plt

env = ConnectFour('rgb_array')
state, info = env.reset()

n_iterations = st.slider('MCTS Search Iterations', 1, 1000, 800)

if 'terminated' not in st.session_state:
    st.session_state['terminated'] = False
    terminated = False
else:
    terminated = st.session_state.terminated

if 'board_state' not in st.session_state:
    st.session_state['board_state'] = env.board_state
else:
    env.board_state = st.session_state.board_state

if 'current_player' not in st.session_state:
    st.session_state['current_player'] = env.current_player
else:
    env.current_player = st.session_state.current_player

# Select Player
if 'player' not in st.session_state:
    options = [1, 2]
    player = st.selectbox('Select Player', options, index=None)

    if not player:
        st.warning('Please input a player')
        st.stop()
    else:
        player = int(player)
        st.session_state['player'] = player
else:
    player = st.session_state['player']

c = np.sqrt(2)
steps = 0

if not terminated:
    steps += 1
    col1, col2 = st.columns(2)

    if player == env.current_player:
        # Select action based on Player Input
        possible_actions = env.get_possible_actions()

        with col1:
            action = st.selectbox('Select action (=column index)', possible_actions, index=None)

            if not action and action != 0:
                with col2:
                    fig, ax = plt.subplots()
                    ax.imshow(env.board_state)
                    st.write(fig)
                    #st.text(env.board_state)
                st.warning('Please input an action')
                st.stop()
            else:
                action = int(action)
    else:
        # Select action based on MCTS
        mcts = MCTS(env)
        action = mcts.run_mcts(n_iterations=n_iterations, c=c)

        with col1:
            st.text(f'Action by MCTS {action}')

    next_state, reward, terminated, truncated, info = env.step(action)
    st.session_state['board_state'] = env.board_state
    st.session_state['current_player'] = env.current_player
    st.session_state['terminated'] = terminated

    with col2:
        fig, ax = plt.subplots()
        ax.imshow(env.board_state)
        st.write(fig)
        #st.text(env.board_state)

    # Button for next move
    if not terminated:
        st.button("Next Move", type="primary")

if terminated:
    if reward == 1 or reward == 2:
        st.text(f'Game won by Player: {reward}')
        st.balloons()
    else:
        st.text('Draw')
else:
    st.text('Open Game')