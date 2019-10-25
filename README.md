# robohearts

**Akash Kwatra and Lucas Kabela**

_Using reinforcement learning to beat the game of hearts. <3_

---

### Proposal
Our project team will consist of Lucas Kabela and Akash Kwatra.  Our proposed project is to implement an agent that learns the game of Hearts.  We plan to use starter code from OpenAI Gym (https://github.com/zmcx16/OpenAI-Gym-Hearts) to bootstrap our efforts.  The motivating question of our project is “can a reasonably good RL agent be developed for the game of hearts, and if so, what information from the domain is most essential”. Along these lines, we plan to survey the methods in reinforcement learning - namely parameterized methods such as actor critic and policy gradient and time permitting, self play. We do plan on starting out with a vastly simplified tabular case, serving as a baseline to compare our parameterized agents against, which we expect to greatly outperform the tabular methods, and possibly even humans. 

The game of hearts is a trick-winning game, a “trick” being defined as a collection of four cards won after each player places down a card. In this game, exactly four players are dealt 13 cards and play one round, composed of 13 tricks. The winner of each trick is the player that has placed down the highest card of the leading suit. In the event a player does not have a card of the leading suit, any card may be played (with the exception of the first trick - in which hearts cannot be played). For each heart in a trick, the winner of the trick gets one point. The queen of spades awards 13 points. After 13 tricks, points are counted, cards are dealt, and another round begins. These rounds repeat until one player accumulates 100 points, at which point the player with the fewest points is declared the winner. If a player manages to win every single heart and the queen of spades in one round, they have “Shot the Moon”. Shooting the moon awards 0 points to the winner, and 26 points to all other players. In one round, the player with the 2 of spades leads the first trick, with the winner of the previous trick leading all subsequent tricks. Hearts cannot be used to lead a trick until someone has “broken hearts”, meaning someone has placed down a heart in a previous trick. A final rule is that before each round, players can select three cards from their hand to trade with another player. 

In regards to our agent, we plan to integrate the rules of this game into the action  space,  meaning if an action were to violate the rules of the game, it would not be available to the agent.  Another caveat is that our initial agent will not participate in card trading (we may use a simple rule based algorithm if this is required by the starter code).   We now address the general questions of our reinforcement learning problem

**1. In what sense is the problem sequential?**

* The game of hearts can easily be broken up into logical units of time. In increasing order of granularity, we have the game (which we shall consider to be one episode), the round (or hand), and the trick, which corresponds to a single time step in which the agent will select an action.

**2. What is your problem's state space?**

* Our state would include, at the most: the current score of all the players, cards that have been played, and cards in agent’s hand .  For this reason, we formulate our problem as a POMDP, as the state space (other player’s hands) is partially observable.  Note, for our initial efforts, we plan to greatly simplify this state space to allow for a tabular formulation.

**3. What is your problem's action space?**

* The cards the agent can choose to play, and in our stretch goal, 3 cards to trade compose the action space.  We shall further reduce the action space by restricting the available actions to those which obey the rules of the game.

**4. What reward function will you use?**
* The negative of the score received from playing a card will serve as the reward function, since the lowest score wins.  This means the agents goal should be to reduce the number of points it wins.

**5. What is the simplest possible first result that you will  try to get? What RL algorithm will you use? What will be the baseline you compare against?**
* The simplest goal is for us to develop an RL agent that can do better than random selection.  We first plan to achieve this in the tabular case, then extend to parameterized methods, where we hope to outperform most rule based agents, and even some humans.  From the tabular methods, our most likely candidates are SARSA and Q-learning, while policy parameter methods we plan to investigate are policy gradient, and actor-critic.  Our initial plan for parameterization of the state space is a neural net to serve as our parameterization.

**6. What will be the stretch goal for your project?**
* One of our potential stretch goals is to examine the game performance with the algorithmic complexity.  Thus, we hope to design an agent that can beat other A.I players with the simplest algorithm.  A secondary stretch goal is to produce an agent that can participate in the trading portion of the card game.  We believe this portion to be very important to an optimal strategy, therefore we would be interested in seeing how building a trading component on top of a game playing agent will boost performance.

In conclusion, our team plans to construct agents capable of playing Hearts using tabular and policy parameterization methods, while investigating the computational tradeoff and scoring advantages of each in a multiplayer partially observable environment.  Our agent will focus on play, and will initially not be scoped to trade cards.  We believe this environment and question is an exciting one to ask, as multiplayer card games which have traditionally been explored (ie Poker) involve a great deal of “bluffing”, while hearts relies much more on play strategy than bluffing.  Furthermore, our motivating question investigates the idea of “how much information is truly needed to produce a “good” agent (if one can be produced)?”