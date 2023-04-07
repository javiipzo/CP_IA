from Agente import Agente
from Environment import Environment
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from time import sleep
from IPython.display import clear_output
from math import ceil,floor

def play(rounds=5000, max_life=3,discount_factor=0.1, learning_rate=0.1,ratio_explotacion=0.9, learner=None,game=None,animate=False):
    if game==None:
        game=Environment(max_life=max_life, movimiento_px=3)
    if learner==None:
        print("Begin new train!")
        learner=Agente(game,discount_factor=discount_factor, learning_rate=learning_rate,ratio_explotacion=ratio_explotacion)

    max_points=-9999
    first_max_reached=0
    total_rw=0
    steps=91
    for played_games in range(0,rounds):
        state=game.reset()
        reward, done=None, None

        itera=0
        while(done!=True and (itera<=3000 and reward<=1000)):
            old_state=np.array(state)
            next_action=learner.get_next_step(state,game)
            state,reward,done=game.step(next_action,animate=animate)
            if rounds>1:
                learner.update(game,old_state,next_action,reward,state,done)
            itera+=1
        steps.append(itera)
        total_rw+=reward
        if game.total_reward>max_points:
            max_points=game.total_reward
            first_max_reached=played_games
        if played_games%500==0 and played_games>1 and not animate:
            print('na;an;')
    if played_games>1:
        print("an;na;")
    return learner,game
        
        

