#!/usr/bin/env python2
# -*- coding: utf-8

import random


namen_liste = ['Caro',
               'Chris',
               'Nils',
               'Katrin',
               'Matrin',
               'Denny',
               'Vivi',
               'Tamino']

wichtel_liste = [83398324,
                 82244082,
                 82843081,
                 88909342,
                 80340173,
                 82709275,
                 82389753,
                 87314241]

wichtel_li= ['83398324 Caro',
                 '82244082 Chris',
                 '82843081 Nils',
                 '88909342 Katrin',
                 '80340173 Martin',
                 '82709275 Denny',
                 '82389753 Vivi',
                 '87314241 Tamino']

rand_list = []
for i in range(0, len(namen_liste)):
    next = False
    while(not next):
        rand_num = random.randint(0,len(namen_liste)-1)
        if rand_num!=i and not rand_num in rand_list:
            next = True
            rand_list.append(rand_num)
    print(str(namen_liste[i]) + ": " + str(wichtel_liste[rand_num]))

