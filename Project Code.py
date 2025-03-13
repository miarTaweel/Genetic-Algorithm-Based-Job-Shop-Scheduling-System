# Name1: Leena Abuhammad
# ID2:1211460
# Name2: Miar Taweel
# ID2:1210447

import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
inputt=[]
POPULATION_SIZE=10
due_dates = {}
MAXIMUN_GENERATION=30

# Function to generate Gantt charts for given operations
def generate_gantt_charts(operations):
    # Initialize dictionaries to track machine and job times
    machines = {}
    machine_times = {}
    job_times = {}

    # Iterate over each operation in the list
    for row_index in range(len(operations)):
        # Extract operation details
        operation = operations[row_index]
        job, machine, duration, op = operation

        # Update machine times
        if machine not in machine_times:
            machine_times[machine] = 0

        # Calculate start time for the current job on the machine
        if job in job_times:
            start_time = max(machine_times[machine], job_times[job])
        else:
            start_time = machine_times[machine]

        # Calculate end time for the current job on the machine
        end_time = start_time + duration

        # Update machine and job times
        machine_times[machine] = end_time
        job_times[job] = end_time

        # Add operation to the corresponding machine
        if machine not in machines:
            machines[machine] = []
        machines[machine].append((f'J{job}', start_time, end_time))

    # Sort operations for each machine by start time
    for machine in machines:
        machines[machine].sort(key=lambda x: x[1])

    return machines


# Function to calculate the total tardiness based on completion times and due dates
def calculate_tardiness(completion_times, due_dates):
    # Initialize total tardiness
    total_tardiness = 0

    # Iterate over each job in the completion times
    for job in completion_times.keys():
        # Get the completion time and due date for the job
        completion_time = completion_times[job]
        due_date = due_dates[job]

        # Calculate tardiness for the job and add it to total tardiness
        tardiness = max(0, completion_time - due_date)
        total_tardiness += tardiness

    return total_tardiness


# Function to calculate the completion times for each job based on machine schedules
def calculate_completion_times(machine_schedules):
    # Initialize dictionary to store completion times for each job
    completion_times = {}

    # Iterate over each machine and its schedule
    for machine, schedule in machine_schedules.items():
        # Iterate over each job in the machine's schedule
        for job, start_time, end_time in schedule:
            # Extract the job ID from the job name
            jobID = int(job[1:])

            # Update completion time for the job if needed
            if jobID not in completion_times or end_time > completion_times[jobID]:
                completion_times[jobID] = end_time

    return completion_times


# Function to calculate fitness values for each schedule in the population
def calculate_population_fitnessfun(schedules, due_dates):
    # Initialize list to store fitness values for each schedule
    fitness_function_values = []
    fitness_sum = 0

    # Iterate over each schedule in the population
    for schedule in schedules:
        # Generate Gantt charts for the schedule
        gantt_charts = generate_gantt_charts(schedule)

        # Calculate completion times for the schedule
        completion_times = calculate_completion_times(gantt_charts)

        # Calculate total weighted tardiness for the schedule
        total_tardiness = calculate_tardiness(completion_times, due_dates)

        # Append the total tardiness to the list of fitness values
        fitness_function_values.append(total_tardiness)

        # Accumulate total tardiness to calculate fitness sum
        fitness_sum += total_tardiness

    # Normalize fitness values
    normalized_fitness_values = [round(fitness / fitness_sum, 4) for fitness in fitness_function_values]

    return fitness_function_values


# Function to perform crossover between two parents
def crossover(parent1, parent2):
    # Randomly select crossover point
    crossover_point = random.randint(1, len(parent1) - 2)

    # Split parents at crossover point
    part1_1 = parent1[:crossover_point]
    part2_1 = parent2[:crossover_point]
    part1_2 = parent1[crossover_point:]
    part2_2 = parent2[crossover_point:]

    # Create children by combining parts of parents
    child1 = part1_1 + part2_2
    child2 = part2_1 + part1_2

    return child1, child2
# Function to convert input data to a chromosome representation
def convert_to_chromosome(input_data):
    chromosome = []
    for operations in input_data:
        for operation in operations:
            job_id = operation[0]
            chromosome.append(job_id)
    return chromosome

def map_chromosome_to_operations(chromosome, input_data):
    operation_index = {job[0][0]: 0 for job in input_data}  # Track the operation index for each job
    mapped_operations = []

    for job_id in chromosome:
        job_operations = input_data[job_id - 1]  # Get the operations for the current job
        current_op_index = operation_index[job_id]
        #print(f"current opetaion index:{current_op_index}")
        mapped_operations.append(job_operations[current_op_index])
        operation_index[job_id] += 1  # Move to the next operation for the job

    return mapped_operations

# Function to generate a population of shuffled chromosomes
def generate_population(chromosome, n):
    shuffled_chromosomes = []
    # Add the original chromosome to the population
    shuffled_chromosomes.append(chromosome)
    # Generate n-1 additional shuffled chromosomes
    for _ in range(n-1):
        shuffled_list = chromosome[:]
        random.shuffle(shuffled_list)
        # Check if the shuffled chromosome is unique before adding
        if shuffled_list not in shuffled_chromosomes:
            shuffled_chromosomes.append(shuffled_list)
    return shuffled_chromosomes

# Function to set the fitness threshold based on initial fitness values
def set_fitness_threshold(initial_fitness_values, improvement_factor=2):
    # Find the best initial fitness value
    best_initial_fitness = min(initial_fitness_values)
    # Calculate the fitness threshold as a fraction of the best initial fitness
    fitness_threshold = best_initial_fitness / improvement_factor
    return fitness_threshold

# Function to perform mutation on a child chromosome
def mutation(input, childe):
    x = 0
    # Create a copy of the child chromosome
    child = childe.copy()
    # Convert the input data to a chromosome
    alljobs = convert_to_chromosome(input)
    # Count occurrences of each job in the input chromosome
    count_dict = {}
    for number in alljobs:
        if number in count_dict:
            count_dict[number] += 1
        else:
            count_dict[number] = 1
    # Sort the count dictionary
    c = [(num, count) for num, count in count_dict.items()]
    count = sorted(c, key=lambda x: x[0])
    # Count occurrences of each job in the child chromosome
    count_dict = {}
    for number in child:
        if number in count_dict:
            count_dict[number] += 1
        else:
            count_dict[number] = 1
    # Sort the count dictionary for the child chromosome
    c_child = [(t[0], count_dict.get(t[0], 0)) for t in count]

    # Equalize the lengths of the count dictionaries
    count_child = sorted(c_child, key=lambda x: x[0])
    if len(count) < len(count_child):
        count += [(0, 0)] * (len(count_child) - len(count))
    elif len(count) > len(count_child):
        count_child += [(0, 0)] * (len(count) - len(count_child))

    # Calculate the validity of the child chromosome
    Validity = [(t1[0], t2[1] - t1[1]) for t1, t2 in zip(count, count_child)]

    # Perform mutation
    while True:
        # Find the first positive and negative tuples
        first_positive_tuple = None
        first_negative_tuple = None
        for i, t in enumerate(Validity):
            if t[1] > 0:
                new_tuple = (t[0], t[1] - 1)
                Validity[i] = new_tuple
                first_positive_tuple = t[0]
                break

        for i, t in enumerate(Validity):
            if t[1] < 0:
                new_tuple = (t[0], t[1] + 1)
                Validity[i] = new_tuple
                first_negative_tuple = t[0]
                break
        # If no negative tuple, break the loop
        if not first_negative_tuple:
            break;

        # Find indices of first positive tuple in the child chromosome
        indices = [i for i, value in enumerate(child) if value == first_positive_tuple]

        # If indices found, randomly select an index and perform mutation
        if indices:
            random_index = random.choice(indices)
            child[random_index] = first_negative_tuple
            x = 1

    return child, x
def convert_file_to_list_of_lists(file_path):
    # Read the file contents
    with open(file_path, 'r') as file:
        input_string = file.read().strip()

    # Split the string into individual job lines
    job_lines = input_string.split('\n')

    # Initialize the list to store the result
    result = []

    # Process each job line
    for j,job_line in enumerate(job_lines):
        # Extract job number
        job_number = j+1
        # Extract tuples from the job line
        tuples = job_line.split(':')[1].split(' -> ')

        # Initialize list to store tuples for the current job
        job_tuples = []

        # Process each tuple
        for i, tuple_str in enumerate(tuples):
            # Extract machine number and number in brackets
            machine_number = int(tuple_str.split('[')[0][1:])
            number_in_brackets = int(tuple_str.split('[')[1].split(']')[0])

            # Create tuple with job number, machine number, number in brackets, and position
            job_tuples.append((job_number, machine_number, number_in_brackets, i + 1))

        # Append tuples for the current job to the result
        result.append(job_tuples)

    return result





def select_parent(population, fitness_values, k):
        # Sort the population indices based on fitness values in ascending order
        sorted_indices = sorted(range(len(fitness_values)), key=lambda x: fitness_values[x])

        # Select the best k fitness values and their corresponding chromosomes
        best_indices = sorted_indices[:k]
        best_chromosomes = [population[i] for i in best_indices]
        parent1, parent2 = random.sample(best_chromosomes, 2)

        # Get the indices of the selected parents
        parent1_index = population.index(parent1)
        parent2_index = population.index(parent2)

        # Print the fitness values of the selected parents
        #print(fitness_values)
        #print("Fitness of Parent 1:", fitness_values[parent1_index])
        #print("Fitness of Parent 2:", fitness_values[parent2_index])

        return parent1, parent2


# Function to generate the genetic algorithm
def generate_genetic_algorithm():
    # Map the population to operations
    mapped_population = []
    # Convert the input data to chromosome
    grandparent_chromosome = convert_to_chromosome(inputt)
    # Generate the initial population
    population = generate_population(grandparent_chromosome, POPULATION_SIZE)

    # Map each chromosome to operations
    for chromosome in population:
        mapped_operations = map_chromosome_to_operations(chromosome, inputt)
        mapped_population.append(mapped_operations)

    # Calculate fitness values for the population
    fitness_values = calculate_population_fitnessfun(mapped_population, due_dates)
    # Set fitness threshold
    fitness_threshold = set_fitness_threshold(fitness_values, improvement_factor=2)
    best_chromosome_index = -1

    # Define the selection size
    k = POPULATION_SIZE // 2
    count=0
    # Iterate until maximum generation
    while len(population) < MAXIMUN_GENERATION:

        # Select parents for crossover
        p1, p2 = select_parent(population, fitness_values, k)

        # Perform crossover
        child1, child2 = crossover(p1, p2)

        # Perform mutation on children
        child1_new, x1 = mutation(inputt, child1)
        child2_new, x2 = mutation(inputt, child2)

        # Add children to the population if they are not already present
        if child1_new not in population:
            population.append(child1_new)
            mapped_operations1 = map_chromosome_to_operations(child1_new, inputt)
            mapped_population.append(mapped_operations1)

        if child2_new not in population:
            population.append(child2_new)
            mapped_operations2 = map_chromosome_to_operations(child2_new, inputt)
            mapped_population.append(mapped_operations2)

        # Recalculate fitness values
        fitness_values = calculate_population_fitnessfun(mapped_population, due_dates)
        best_chromosome_fitness = min(fitness_values)
        best_chromosome_index = fitness_values.index(best_chromosome_fitness)

        # Check if the best chromosome's fitness value meets the threshold
        if best_chromosome_fitness <= fitness_threshold:
            print(f"The solution fitness values: {best_chromosome_fitness}, threshold: {fitness_threshold}")
            return population[best_chromosome_index]

    # Print the solution fitness values and threshold
    print(f"The solution fitness values: {best_chromosome_fitness}, threshold: {fitness_threshold}")
    return population[best_chromosome_index]


POPULATION_SIZE = int(input("Enter the population size: "))

# Prompt the user to enter the maximum generation
MAXIMUN_GENERATION = int(input("Enter the maximum generation: "))

#Read input from a text file
file_path = 'text.txt'
inputt = convert_file_to_list_of_lists(file_path)


#Calculate due date of all jobs in the input
for job in inputt:
    job_id = job[0][0]
    total_duration = sum(operation[2] for operation in job)
    due_dates[job_id] = total_duration
# Start generating the Algorithm
solution=generate_genetic_algorithm()
solution=map_chromosome_to_operations(solution,inputt)
solution_ganttchart=generate_gantt_charts(solution)

#Print the schedual
print("Ganttchart for the optimal schedule:")
for machine, operations in solution_ganttchart.items():
           print(f'Machine {machine}:')
           for operation in operations:
                print(operation)
           print("___________________________________________________")
#plot the GanttChart of the optimal schedual
job_schedule = solution_ganttchart


#__________________________________________________________________________________________________________
# Create a figure and axis
fig, ax = plt.subplots(figsize=(15, 6))
# Colors for each job
colors = ['#E6E6FA', '#EE82EE', '#800080', '#9370DB', '#4B0082', '#9932CC', '#ffd1b3', '#ffb39f', '#b3ffb3', '#b3b3ff', '#ffb3d4', '#b3ffff', '#ffd1b3']
color_map = {}

# Function to generate random pastel colors
def generate_pastel_color():
    r = (random.randint(0, 255) + 255) // 2
    g = (random.randint(0, 255) + 255) // 2
    b = (random.randint(0, 255) + 255) // 2
    return f'#{r:02X}{g:02X}{b:02X}'

# Plot each machine's job schedule

for machine, jobs in job_schedule.items():
    for job in jobs:
        job_name, start, end = job
        if job_name not in color_map:
            if len(color_map) >= len(colors):
                # Generate a new color and add it to the colors array
                new_color = generate_pastel_color()
                colors.append(new_color)
            color_map[job_name] = colors[len(color_map) % len(colors)]
        ax.broken_barh([(start, end - start)], (machine - 0.4, 0.8), facecolors=(color_map[job_name]))

# Set labels
ax.set_xlabel('Time', fontname='Times New Roman')
ax.set_ylabel('Machines', fontname='Times New Roman')

# Dynamically create y-ticks and labels based on the job_schedule keys
machines = sorted(job_schedule.keys())
ax.set_yticks(machines)
ax.set_yticklabels([f'Machine {machine}' for machine in machines], fontname='Times New Roman')
ax.grid(True)

# Adjust x-axis to show smaller time units
# Adjust x-axis to show larger time units
ax.set_xlim(0, max(max(end for job in jobs for _, _, end in job_schedule[machine]) for machine in job_schedule))
ax.xaxis.set_major_locator(plt.MultipleLocator(3))  # Set major ticks at every 5 time units
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))  # Set minor ticks at every 1 time units
ax.tick_params(which='both', length=5)  # Adjust tick length


# Create a legend
patches = [mpatches.Patch(color=color, label=job_name) for job_name, color in color_map.items()]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot with title in Times New Roman and bold
plt.title('Job Scheduling on Machines', fontname='Times New Roman', fontweight='bold')
plt.show()