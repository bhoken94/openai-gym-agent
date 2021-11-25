# Deep Q-Learning Agent in PyTorch
> Quella che segue è l'implementazione di un Agente in pyTorch, usando la libreria OpenAI Gym per il setup dell'enviroment. L'agente usa come policy la e-greedy.
> Implementazione presente su Git:
> - https://github.com/bhoken94/openai-gym-agent
> - https://github.com/bhoken94/ai-self-driving-car-kivy

## Definire la network
Iniziamo con importare le librerie che ci servono. Usiamo un approccio OOP, quindi creeremo la classe Network dove:
```python
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  
  
class Network(nn.Module):  
	def __init__(self, input_size, num_action, lr):  
 		super(Network, self).__init__()  
 		self.input_size = input_size  
 		self.num_action = num_action  
 		self.fullConnection1 = nn.Linear(input_size, 264)  
 		self.fullConnection2 = nn.Linear(264, 264)  
 		self.fullConnection3 = nn.Linear(264, num_action)  
 		self.optimizer = optim.Adam(self.parameters(), lr=lr)  
  
 	# Implementiamo la forward propagation dove mandiamo l'input agli hidden layer  
 	def forward(self, state):  
 		x = F.relu(self.fullConnection1(state))  
 		x = F.relu(self.fullConnection2(x))  
 		output = self.fullConnection3(x)  
		return output
```
la classe Network eredita molti metodi dalla classe Module, fornita da pyTorch. All'interno del costruttore, definiamo:
- **input_size** : numero dei input (segnali) che l'agente percepisce dall'enviroment
- **num_action** : numero delle azioni che può intraprendere l'agente
- **fullConnection** : quanti strati di hidden layer vogliamo dare alla nostra rete. L'ultimo strato e il primo devono avere rispettivamente num_action come ultimo parametro e input_size come primo parametro.
- **optimizer** : definiamo infine l'optimizer da assegnare alla rete e gli passiamo il learning rate.

## Definire Experience Replay
Una volta definita la nostra rete, dobbiamo definire la memoria dei ricordi, che l'agente userà per imparare dai propri errori. Iniziamo definendo la classe Experience Replay come segue:
```python
import numpy as np  
import torch  
  
  
class ExperienceReplayMemory(object):  
 	def __init__(self, capacity, input_dims):  
 		self.capacity = capacity  
 		self.state_memory = np.zeros((self.capacity, *input_dims),  
                                     dtype=np.float32)  
 		self.new_state_memory = np.zeros((self.capacity, *input_dims),  
                                         dtype=np.float32)  
 		self.action_memory = np.zeros(self.capacity, dtype=np.int32)  
 		self.reward_memory = np.zeros(self.capacity, dtype=np.float32)  
 		self.done_memory = np.zeros(self.capacity, dtype=np.bool)  
 		self.memory_counter = 0  
  
 	def push(self, last_state, action, reward, new_state, done):  
 		index = self.memory_counter % self.capacity  
        self.state_memory[index] = last_state  
 		self.new_state_memory[index] = new_state  
 		self.reward_memory[index] = reward  
 		self.action_memory[index] = action  
 		self.done_memory[index] = done  
  
 		self.memory_counter += 1  
  
	# Metodo per ritornare un batch di transizioni dalla memoria per aggiornare la rete  
 	def sample(self, batch_size):  
 		max_mem = min(self.memory_counter, self.capacity)  
 		batch = np.random.choice(max_mem, batch_size, replace=False)  
 		state_batch = torch.tensor(self.state_memory[batch])  
 		new_state_batch = torch.tensor(self.new_state_memory[batch])  
 		action_batch = self.action_memory[batch]  
 		reward_batch = torch.tensor(self.reward_memory[batch])  
 		terminal_batch = torch.tensor(self.done_memory[batch])  
 		return state_batch, new_state_batch, action_batch, reward_batch, terminal_batch
```
Nel costruttore della classe definiamo i parametri:
- **capacity** : ovvero quanti ricordi può conservare l'agente. Ogni volta che la mempria raggiunge la sua massima capacità, i vecchi ricordi vengono rimpiazzati dai nuovi. Infatti: `index = self.memory_counter % self.capacity`, che rappresenta l'indice che vogliamo assegnare ad ogni nuovo ricordo, sarà 0 quando il resto della operazione tra il counter e la capacità sarà appunto 0.
- **state_memory, new_state_memory, action_memory, reward_memory, done_memory** : definiamo gli array bidimensionali (per ora popolati da 0), dove numero delle righe=capacità della memoria e numero colonne = numero degli input che riceve l'agente (compreso done nel caso della libreria OpenAI Gym). Definiamo per ognuno di loro anche il timpo corretto (dtype).
- **memory_counter**: valore che ci servirà per tenere traccia di quanti ricordi sono stati inseriti nella memoria ed ottenere cosi nuovi index (vedi sopra)

In questa classe, definiamo due metodi:
- **push**: questo metodo prende in entrata i valori che comporranno il nostro ricordo o *transizione*, ovvero : ultimo stato, azione, reward, nuovo stato, done. In questo metodo andiamo a ricavarci l'indice dove posizionare i ricordi nelle varie memorie dedicate e li aggiungiamo. Infine incrementiamo il memory counter di 1.
- **sample**: questo metodo, ritorna un batch (cioè un campione) di ricordi, della dimensione di *batch_size*. Definiamo prima la *max_mem*, che viene calcolata prendendo il minimo tra il memory counter e la capacità della memoria. Dopodichè il metodo `np.random.choice()` genererà un array di dimensione uguale alla batch_size, e con valori casuali tra 0 e *max_mem*. Questo array generato rappresenta le "righe" che selezioneremo dalle nostre memorie. Selezioneremo, quindi i nostri batch dalle rispettive memorie bidimensionali e li ritorniamo.

## Definire l'agente
Definiamo ora la classe Agent, dove ci occuperemo di implementare la logica del nostro agente. In questo caso, implementeremo la policy ==epsilon-greedy==:
```python
import torch  
import torch.nn.functional as F  
from egreedy.experience_replay import ExperienceReplayMemory  
from network import Network  
import numpy as np  
  
  
class Agent:  
 	def __init__(self, num_input, num_action, gamma, lr, batch_size, epsilon, eps_min=0.05, eps_dec=5e-4):  
 		self.gamma = gamma  
 		self.epsilon = epsilon  
 		self.epsilon_min = eps_min  
 		self.epsilon_dec = eps_dec  
 		self.batch_size = batch_size  
 		self.lr = lr  
 		self.network = Network(num_input, num_action, lr)  
 		self.reward_window = []  
 		self.memory = ExperienceReplayMemory(capacity=100000, input_dims=[num_input])  
  
 	def select_action(self, observation):  
 		# e-greedy  
 		if np.random.random() > self.epsilon:  
 			state = torch.tensor([observation])  
 			actions = self.network.forward(state)  
 			action = torch.argmax(actions).item()  
 		else:  
 			action = np.random.choice(self.network.num_action)  
 		return action  
 
 	def store_transition(self, new_state, action, reward, last_state, done):  
 		self.memory.push(last_state, action, reward, new_state, done)  
  
 		# Funzione che prende in entrata i batch degli stati, azioni e reward  
 	def learn(self):  
 		if self.memory.memory_counter < self.batch_size:  
 			return  
 		self.network.optimizer.zero_grad() # resetto i gradienti dell'optimizer  
 		batch_index = np.arange(self.batch_size, dtype=np.int32)  
 		state_batch, new_state_batch, action_batch, reward_batch, done_batch = self.memory.sample(self.batch_size)  
 		q_state = self.network.forward(state_batch)[batch_index, action_batch] # ritorna i q value per quel batch_action  
 		q_state_next = self.network.forward(new_state_batch)  
 		q_state_next[done_batch] = 0.0  
 		q_target = self.gamma * torch.max(q_state_next, dim=1)[0] + reward_batch  
        td_loss = F.smooth_l1_loss(q_state, q_target) # calcolo il loss  
 		td_loss.backward(retain_graph=True) # aggiorno i gradienti  
 		self.network.optimizer.step() # aggiorno le weights  
 		# e-greedy 
		self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
```
Il costruttore della classe Agent prende in entrata i seguenti:
- **num_input** : numero dei segnali che l'agente riceve
- **num_action**: numero di azioni che può eseguire l'agente
- **gamma** : valore che rappresenta il ==discount factor== che viene applicato nel calcolo del target, ovvero il nuovo punto di arrivo della rete.
- **lr**: valore del learning rate
- **batch_size**: dimensione che vogliamo dare quando inviamo i batch delle memorie alla rete per imparare da esse.
- **epsilon** : valore della tua epsilon per la epsilon-greedy policy
- **eps_min** : valore minimo di epsilon
- **eps_dec**: valore da togliere ad epsilon

Dopodichè nel costruttore definiamo anche una istanza della classe Network, passandogli i valori corrispondenti e una istanza della classe ExperienceReplayMemory.
Nella classe Agent definiziamo poi i seguenti metodi:
- **select_action**: prende in entrata una observation che contiene : last_state, action, reward, new_state e done. Qui applichiamo la epsilon greedy policy: se un numero random è maggiore del nostro valore epsilon, selezioniamo un azione passando alla rete la nuova observation, convertita in Tensor, e passiamo il risultato in una funzione argmax, che ritorna gli indici del valore più alto presente al suo interno; altrimenti slezioniamo una azione casuale dalla num_action. Infine ritorniamo il risultato
- **store_transition**: prende in entrata i valori che compongono la observation e li aggiungiamo alla memoria dell'agente
- **learn**: metodo che si occupa del training vero e proprio della rete. Per prima cosa, verifichiamo che la memoria abbia abbastanza memorie rispetto alla batch_size. Dopodichè resettiamo i gradienti dell'optimizer. Una volta fatto ciò, creeiamo un array contenente i valori da 0 a batch_size, che ci serviranno, insieme al batch delle azioni per selezionare i valori di q, una volta passato il batch dei last state nella network. Si calcola poi il valore di q per i new_state batch, passandolo nel metodo forward della rete. Una volta calcolati questi due valori, settiamo nel batch dei done, contenuto all'interno di q_state_next, il valore 0.0 e calcoliamo poi il target, moltiplicando il gamma per il massimo dei valori di q. Infine, calcoliamo la loss, passando nella funzione L1, i valori di q del last state e il target. Poi passiamo la loss nella rete, aggiorniamo le weight dell'optimizer e ricalcoliamo il valore di epsilon.