import csv
import json
import matplotlib.pyplot as plt

csv_file = open('fitness_history.csv', 'r')
events_file = open('generation_events.json', 'r')
events_dict = json.load(events_file)

invalid_events_generation = []
coin_collected_event_generation = []
 
# creating dictreader object

# creating empty lists
mean_fitness = []
best_fitness = []
generations = []

# iterating over each row and append values to empty list
for best, mean in csv.reader(csv_file, delimiter=' '):
    mean_fitness.append(float(mean))
    best_fitness.append(float(best))

for key in events_dict:
    invalid_events_generation.append(int(events_dict[key]['INVALID_ACTION']))
    if 'COIN_COLLECTED' in list(events_dict[key].keys()):
        coin_collected_event_generation.append(int(events_dict[key]['COIN_COLLECTED']))
    else:
        coin_collected_event_generation.append(0)

generations = [i for i in range(len(best_fitness))]

figure, axis = plt.subplots(1,3)

figure.set_size_inches(15,5)

axis[0].plot(generations, best_fitness)
axis[0].set_title('Best Fitness')
axis[0].set_ylim([min(best_fitness), max(best_fitness)])
axis[1].plot(generations, mean_fitness)
axis[1].set_title('Mean Fitness')
axis[2].plot(generations, invalid_events_generation, color = 'red', label = "Invalid Actions")
axis[2].plot(generations, coin_collected_event_generation, color = 'green', label = "Coin Collected")
axis[2].legend(loc = 'upper right')
axis[2].set_title('Events')
#figure.suptitle('Stastics for NEAT_FEATSEL agent', fontsize=16)

plt.show()



