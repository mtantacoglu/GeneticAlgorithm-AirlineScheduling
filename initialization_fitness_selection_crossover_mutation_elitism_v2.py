

import pandas as pd
import numpy as np
import random
import copy

# Aircraft types and capacities
aircraft_df = pd.DataFrame({
    'Aircraft Number': [1, 2, 3],
    'Capacity': [50, 60, 70]
})

# Arrivals per day for one week
arrivals_df_list = [
    pd.DataFrame({'Hour': np.arange(1, 13), 'Passengers': np.random.randint(1, 26, size=12)}),
    pd.DataFrame({'Hour': np.arange(1, 13), 'Passengers': np.random.randint(1, 26, size=12)}),
    pd.DataFrame({'Hour': np.arange(1, 13), 'Passengers': np.random.randint(1, 26, size=12)}),
    pd.DataFrame({'Hour': np.arange(1, 13), 'Passengers': np.random.randint(1, 26, size=12)}),
    pd.DataFrame({'Hour': np.arange(1, 13), 'Passengers': np.random.randint(1, 26, size=12)}),
    pd.DataFrame({'Hour': np.arange(1, 13), 'Passengers': np.random.randint(1, 26, size=12)}),
    pd.DataFrame({'Hour': np.arange(1, 13), 'Passengers': np.random.randint(1, 26, size=12)})
]

#waiting cost parameters
Beta0=19
Beta1=8
Beta2=10
Beta3=34
Ta0=0.9
Ta1=1.2
Ta2=2
B0=0
B1=3
B2=7
B3=10

max_waiting_time=8

def calculate_waiting_cost(waiting):
    if ((B0<waiting) and (waiting<=B1)):
        return Beta0+Ta0*(waiting-B0)
    elif ((B1<waiting) and (waiting<=B2)):
        return Beta1+Ta1*(waiting-B1)
    elif ((B2<waiting) and (waiting<=B3)):
        return Beta2+Ta2*(waiting-B2)
    else:
        return 0
    
    
def generate_initial_solutions(num_solutions,arrivals_df_list,aircraft_df,max_waiting_time): 

    #Total arrival number list per day
    arrival_day=[]
    for i in arrivals_df_list:
        arrival_day.append(i['Passengers'].sum()) 
    
    # Define the time slots (0-4, 4-8, 8-12)
    time_slots = [(1, 4), (5, 8), (9, 12)]
    
    #Solution list to store solutions
    all_solutions = []
    
    for solution_num in range(num_solutions):

        #Initializaing key variables in each solution
        carryover_df = pd.DataFrame(columns=['Hour', 'Passengers'])        
        total_flights = []        
        carryover_list = []
        spilled_list=[]
        total_day_list=[]    
        previous_carryover = 0
        previous_carryover_list=[] 
        day_total_waiting_time_list = []
        day_total_waiting_cost_list = []        
        track_combined_arrivals=[]
        
        # 7 günlük plan yapalım
        for day in range(7):
            
            #Reseting variables at the beginning of the each day
            carryover_day = 0  
            spilled_day=0
            total_assigned_day=0
            combined_arrivals=0
            day_total_waiting_time=0
            day_total_waiting_cost=0
            
            # Copy arrivals for the corresponding day
            arrivals_df = arrivals_df_list[day].copy()

            # Index adjustment if there is a carry over from previous day for carryovers and arrivals
            if not carryover_df.empty:
                carryover_df['Hour'] = carryover_df['Hour'] - 6
                rows_to_drop = carryover_df[carryover_df["Hour"] < 0].index
                carryover_df.drop(rows_to_drop, inplace=True)
                arrivals_df['Hour'] = arrivals_df['Hour'] + 6
                    
            # Merge previous day carryovers and arrivals for the day
            combined_arrivals = pd.concat([carryover_df, arrivals_df]).sort_values(by='Hour', ascending=True).reset_index(drop=True)
        
           
            for slot in time_slots:
                start_time, end_time = slot
                # Randomly select aircraft type for rach slot
                selected_aircraft = aircraft_df.sample(1).iloc[0]
                aircraft_capacity = selected_aircraft['Capacity']
                aircraft_number=selected_aircraft['Aircraft Number']
        
        
                # Randomly select aircraft type for each slot
                opening_hour = np.random.randint(start_time, end_time + 1)
                adjusted_opening_hour = opening_hour
                actual_opening_hour=opening_hour
                
                #Adjut opening hour if there is a carryover from previous day
                if not carryover_df.empty:
                    adjusted_opening_hour = opening_hour + 6
                else:
                    adjusted_opening_hour = opening_hour
                                    
                # Reset 'waiting_time' and 'waiting_cost' columns to 0 at the beginning of each loop iteration
                combined_arrivals['waiting_time'] = 0
                combined_arrivals['waiting_cost'] = 0
                
                total_waiting_time=0
                total_waiting_cost=0
                        
                #Calculate waiting time for passenegrs in the combined arrivals (all arrivals from today+unassigned previous day)
                combined_arrivals["waiting_time"]=adjusted_opening_hour-combined_arrivals["Hour"]
                # Ensure that any negative waiting time is set to zero
                combined_arrivals['waiting_time'] = combined_arrivals['waiting_time'].apply(lambda x: x if x > 0 else 0)
                combined_arrivals["waiting_cost"]=combined_arrivals["waiting_time"].apply(calculate_waiting_cost)
                
                #Generate a set of passengers that are eligible till the aircraft opening and fit max waiting time criteria
                today_passengers = combined_arrivals[(combined_arrivals['Hour'] < adjusted_opening_hour) & (combined_arrivals['waiting_time'] < max_waiting_time)]
                #Find these eligible passngers sum
                total_passengers = today_passengers['Passengers'].sum()
                
                #if eligible passngers sum less or equal to the aircraft capacity, then assign all of them
                if total_passengers <= aircraft_capacity:
                    
                    #track assigned passengers
                    total=total_passengers
                    total_assigned_day += total_passengers
                    # Calculate total waiting time
                    a=combined_arrivals.copy()
                    total_waiting_time  = (a["waiting_time"] * a["Passengers"]).sum()
                    day_total_waiting_time= day_total_waiting_time + total_waiting_time 
                    #Calculate total waiting cost
                    total_waiting_cost = (a["waiting_cost"] * a["Passengers"]).sum()
                    day_total_waiting_cost=day_total_waiting_cost+ total_waiting_cost
                    
                    total_flights.append({
                        "Solution": solution_num + 1,
                        'Day': day + 1,
                        'Slot': f'{start_time}-{end_time}',
                        'Opening Hour': actual_opening_hour,
                        'Aircraft Capacity': selected_aircraft['Capacity'],
                        "Aircraft Number":aircraft_number,
                        "Total passengers": arrivals_df['Passengers'].sum(),
                        'Assigned Passengers': total,
                        "Total waiting cost":total_waiting_cost,
                        "Total waiting time":total_waiting_time 
                    })
                    
                    #delete all passengers in the combined arrivals since they are all assigned
                    rows_to_drop = combined_arrivals[combined_arrivals['Hour'] < adjusted_opening_hour].index
                    combined_arrivals.drop(rows_to_drop, inplace=True)

                    
                else:
                    #Generate a set of passengers that are eligible till the aircraft opening and fit max waiting time criteria
                    remaining_passengers = combined_arrivals[(combined_arrivals['Hour'] < adjusted_opening_hour) & (combined_arrivals['waiting_time'] < max_waiting_time)]
                    #Sort them based on arrival time in ascending order(gelişi ilk olan ilk değerlendirilcek)
                    remaining_passengers_sorted = remaining_passengers.sort_values(by='Hour')
                    
                    #Assigned passenger list under this else
                    passengers_to_board = []  #assigned passenger list
                    total = 0 # initialize zero to track total assigned
                    
                    #for waiting time and cost calculation
                    total_waiting_time=0
                    total_waiting_cost=0
                    
                    combined_arrivals.reset_index(drop=True, inplace=True)
                    
                    # Initialize list to store conditions to delete from combined_arrivals
                    rows_to_delete_from_combined = []
                    
                    #copy combined_arrivals
                    copy_combined_arrivals_df=combined_arrivals.copy()
        
                    for index, row in remaining_passengers_sorted.iterrows():
                        
                        if aircraft_capacity >= row['Passengers']:
                            # Track the row in combined_arrivals by matching 'Hour' and 'Passengers' and delete them
                            condition = (combined_arrivals['Hour'] == row['Hour']) & (combined_arrivals['Passengers'] == row['Passengers'])
                            matched_row_index = combined_arrivals[condition].index
                            if not matched_row_index.empty:
                                #delete from combined_df
                                rows_to_delete_from_combined.append(matched_row_index[0])
                                #calculate waiting cost and time
                                total_waiting_time=(copy_combined_arrivals_df.loc[matched_row_index, 'waiting_time']*copy_combined_arrivals_df.loc[matched_row_index, 'Passengers']).sum()                        
                                total_waiting_cost=(copy_combined_arrivals_df.loc[matched_row_index, 'waiting_cost']*copy_combined_arrivals_df.loc[matched_row_index, 'Passengers']).sum()
                            #update reamining paasengers and aircraft capacity
                            passengers_to_board.append(row)
                            aircraft_capacity -= row['Passengers']
                            total += row['Passengers']
                            remaining_passengers_sorted.drop(index, inplace=True)
        
                            
                            
                        elif aircraft_capacity > 0:
                            
                            # Adjust row for partial assignment in combined_arrivals
                            condition = (combined_arrivals['Hour'] == row['Hour']) & (combined_arrivals['Passengers'] == row['Passengers'])
                            matched_index = combined_arrivals[condition].index[0]
                            remaining_capacity = row['Passengers'] - aircraft_capacity
                            combined_arrivals.at[matched_index, 'Passengers'] = remaining_capacity
                            
                            #calculate waiting cost and waiting time
                            total_waiting_time=(copy_combined_arrivals_df.loc[matched_index,'waiting_time']*copy_combined_arrivals_df.loc[matched_index,'Passengers']).sum() 
                            total_waiting_cost=(copy_combined_arrivals_df.loc[matched_index,'waiting_cost']*copy_combined_arrivals_df.loc[matched_index,'Passengers']).sum() 
                            
                            #Partial assignment
                            row['Passengers'] = row['Passengers'] - aircraft_capacity
                            passengers_to_board.append(aircraft_capacity)
                            total += aircraft_capacity
                            aircraft_capacity = 0
                            remaining_passengers_sorted.at[index, 'Passengers'] -= aircraft_capacity
                            
                        else:  # capacity is full
                            aircraft_capacity = 0
                            break
                    
                    # Drop the rows from combined_arrivals for the all passengers in same interval assigned case(first if)
                    combined_arrivals.drop(rows_to_delete_from_combined, inplace=True)

                    # Add to total boarded passengers for the day (assigned)
                    total_assigned_day += total  
                    
                    #Add to totaltotal waiting cost and time for the day 
                    day_total_waiting_time += total_waiting_time
                    day_total_waiting_cost += total_waiting_cost                    
                                        
                    # Uçağa binen yolcuları ekliyoruz
                    if passengers_to_board:
                        total_flights.append({
                            "Solution": solution_num + 1,
                            'Day': day + 1,
                            'Slot': f'{start_time}-{end_time}',
                            'Opening Hour': actual_opening_hour,
                            'Aircraft Capacity': selected_aircraft['Capacity'],
                            "Aircraft Number":aircraft_number,
                            "Total passengers": arrivals_df['Passengers'].sum(),
                            'Assigned Passengers': total,  # total assigned, 
                            "Total waiting cost":total_waiting_cost,
                            "Total waiting time":total_waiting_time 
                        })
                        
                      
            #used to calculate spilled and carryover in a day for each time slot
            track_combined_arrivals.append(combined_arrivals)
            
            #Tracking carryovers
            #if there are carryovers from previous day
            if not carryover_df.empty:
            # Candidates for carryover to next day if combined_arrivals has hours greater than 12
                carryover_candidates = combined_arrivals[combined_arrivals['Hour'] >=12]
            else:
            # Candidates for carryover from combined arrivals if hours are greater than 6
                carryover_candidates = combined_arrivals[combined_arrivals['Hour'] >=6]
            # Reset carryover_df to include new carryover candidates, and handle empty case gracefully
            if not carryover_candidates.empty:
                carryover_df = carryover_candidates.reset_index(drop=True)
            else:
                carryover_df = pd.DataFrame(columns=['Hour', 'Passengers',"waiting_time","waiting_cost"])  # Reset to an empty DataFrame with correct columns        
            # Sum up carryover passengers for the day
            carryover_day = carryover_df["Passengers"].sum()        
            # At the end of the day, append the total carryover passengers
            carryover_list.append(carryover_day)
            
            #Calculate total assigned for each day
            total_day_list.append(total_assigned_day)
            
            #Calculate total waiting time and cost for day
            day_total_waiting_time_list.append(day_total_waiting_time)
            day_total_waiting_cost_list.append(day_total_waiting_cost)
            
            if day == 0:
                previous_carryover=0
            else:
                previous_carryover=carryover_list[day-1]
            previous_carryover_list.append(previous_carryover)   
                 
            #At the end of the day, append the spilled passengers
            spilled_day= max(arrivals_df['Passengers'].sum() + previous_carryover - total_assigned_day - carryover_df["Passengers"].sum(), 0)
            spilled_list.append(spilled_day)
        """                           
        summary_df=pd.DataFrame({
            "Day":list(range(1,8)),
            "Total Passengers":arrival_day,
            "Carryover Previous Day":previous_carryover_list,
            "Assigned":total_day_list,
            "Carryover Today":carryover_list,
            "Spilled":spilled_list,
            "Day total waiting cost":day_total_waiting_cost_list,
            "Day total waiting time":day_total_waiting_time_list,
            
            })
        """
        # Write total flights results to a dataframe
        final_schedule_df = pd.DataFrame(total_flights)
               
        #Find carryover and spilled for each slot and day        
        results_list = []
        
        for index, i in enumerate(track_combined_arrivals):
            
            if (i['Hour'] > 12).any():    
            
            # Process the first slot (1-10 hours)
                spilled_first_phase = i[(i['Hour'] >= 1) & (i['Hour'] < 10)]['Passengers'].sum()
            
            # Process the second slot (10-12 hours)
                spilled_second_phase = i[(i['Hour'] >= 10) & (i['Hour'] < 12)]['Passengers'].sum()
        
            # Process the third slot (greater than 12 hours, considered spilled)
                spilled_third_phase = 0
                
            # Process the third slot (greater than 12 hours, considered spilled)
                carried_first_phase = 0
                
            #carried passengers
                carried_second_phase = i[(i['Hour'] >= 12) & (i['Hour'] <= 14)]['Passengers'].sum()
            #carried passengers
                carried_third_phase = i[(i['Hour'] > 14) & (i['Hour'] <= 18)]['Passengers'].sum()
                
            else:
                
                # Process the first slot (1-10 hours)
                spilled_first_phase = i[(i['Hour'] >= 1) & (i['Hour'] < 5)]['Passengers'].sum()
            
            # Process the second slot (10-12 hours)
                spilled_second_phase = i[(i['Hour'] == 5)]['Passengers'].sum()
        
            # Process the third slot (greater than 12 hours, considered spilled)
                spilled_third_phase = 0
                
            # Process the third slot (greater than 12 hours, considered spilled)
                carried_first_phase = 0
                
            #carried passengers
                carried_second_phase = i[(i['Hour'] >= 6) & (i['Hour'] <= 8)]['Passengers'].sum()
            #carried passengers
                carried_third_phase = i[(i['Hour'] > 8) & (i['Hour'] <= 12)]['Passengers'].sum()
        
            # Append results to the list as a dictionary
            results_list.append({
                "Day": index+1,
                "Slot": "1-4",
                "Carryover": carried_first_phase if not pd.isna(carried_first_phase) else 0,
                "Spilled": spilled_first_phase if not pd.isna(spilled_first_phase) else 0
            })
        
            results_list.append({
                "Day": index+1,
                "Slot": "5-8",
                "Carryover": carried_second_phase if not pd.isna(carried_second_phase) else 0,
                "Spilled": spilled_second_phase if not pd.isna(spilled_second_phase) else 0
            })
        
            results_list.append({
                "Day": index+1,
                "Slot": "9-12",
                "Carryover":carried_third_phase if not pd.isna(carried_third_phase) else 0,
                "Spilled": spilled_third_phase if not pd.isna(spilled_third_phase) else 0
            })
        
        # Write results_list to a DataFrame
        final_results_df = pd.DataFrame(results_list)
       
        # Merge final_results_df and final_schedule_df based on the columns 'Day' and 'Slot'
        chromosome_df = pd.merge(final_schedule_df, final_results_df, on=['Day', 'Slot'], how='left')
                
        all_solutions.append(chromosome_df)
                        
    return all_solutions

def fitness_function(all_solutions,aircraft_type1_cost,aircraft_type2_cost,aircraft_type3_cost,fare,spilled_cost):
    
    fitness_results=[]
    
    for solution in all_solutions:
        
        solution_number=solution["Solution"].iloc[0]
        
        #Count total used aircraft ragarding aircraft type
        aircraft_type1=solution[solution["Aircraft Number"]==1].shape[0]
        aircraft_type2=solution[solution["Aircraft Number"]==2].shape[0]
        aircraft_type3=solution[solution["Aircraft Number"]==3].shape[0]
        
        #Calculate aircraft cost
        total_aircraft_cost=(aircraft_type1_cost*aircraft_type1)+(aircraft_type2_cost* aircraft_type2)+(aircraft_type3_cost*aircraft_type3)
        
        #Calculate assigned passengers total per day and profit
        total_assigned_passengers = solution['Assigned Passengers'].sum()
        revenue=fare*total_assigned_passengers
        
        #calculate waiting cost
        total_waiting_cost = solution['Total waiting cost'].sum()
        
        #calculate total spilled per day+ day 7 carryover
        total_spilled = solution['Spilled'].sum()
        day_7_carryover = solution[(solution['Day'] == 7)]['Carryover'].sum()
        total_spilled_and_last_day_carryover_number=total_spilled+day_7_carryover
        cost_total_spilled_and_last_day_carryover_number=spilled_cost*total_spilled_and_last_day_carryover_number
        
        fitness_value=revenue-total_aircraft_cost-cost_total_spilled_and_last_day_carryover_number-total_waiting_cost
        
                # Store results for each solution
        fitness_results.append({
            'Solution': solution_number,
            'Total Aircraft Cost': total_aircraft_cost,
            'Total Assigned Passengers': total_assigned_passengers,
            'Total Waiting Cost': total_waiting_cost,
            'Total Spilled': total_spilled,
            'Carryover Day 7': day_7_carryover,
            'Fitness Score': fitness_value
        })

        fitness_results_df=pd.DataFrame(fitness_results)
    return fitness_results_df

def proportional_probability_selection(fitness):
    
    #fitness probabilities are based on the fitness values
    #this step used to probability assignmnet for each solution
    total_fitness_score = fitness["Fitness Score"].sum()
    fitness["Fitness Probability"] = fitness["Fitness Score"] / total_fitness_score

    return fitness

def roulette_wheel_selection(fitness_proportional_selection):
    
    # Calculate cumulative probabilities
    fitness_proportional_selection['Cumulative Probability'] = fitness_proportional_selection['Fitness Probability'].cumsum()
    #Store parents
    parent_solutions=[]
    
    #Select parents by roulette wheel algorithm
    for i in range(2):        
        # Generate a random probability value between 0 and the total cumulative probability
        random_value = random.uniform(0, fitness_proportional_selection['Cumulative Probability'].iloc[-1])        
        # Determine which solution corresponds to the random value
        selected_row = fitness_proportional_selection[fitness_proportional_selection['Cumulative Probability'] >= random_value].iloc[0]        
        # Append the selected solution and its probability to the list
        parent_solutions.append(selected_row['Solution'])
    
    return parent_solutions    
    
def crossover_single_point(parent1,parent2,initial_solutions):
    
    parent1_solution_df = next((df for df in initial_solutions if df['Solution'].iloc[0] == parent1), None)
    if parent1_solution_df is not None:
        # Create a list of lists containing 'Opening Hour' and 'Aircraft Number' information
        genes_parent1 = parent1_solution_df[parent1_solution_df['Solution'] == parent1][['Opening Hour', 'Aircraft Number']].values.tolist()

    parent2_solution_df = next((df for df in initial_solutions if df['Solution'].iloc[0] == parent2), None)
    if parent2_solution_df is not None:
        # Create a list of lists containing 'Opening Hour' and 'Aircraft Number' information
        genes_parent2 = parent2_solution_df[parent2_solution_df['Solution'] == parent2][['Opening Hour', 'Aircraft Number']].values.tolist()
    
    #check whether both parents have same number of genes
    if len(genes_parent1) != len(genes_parent2):
        raise ValueError("The length of genes for both parents must be the same to perform crossover.")   
    
    #determine the crossover point
    crossover_point = random.randint(1, len(genes_parent1) - 1)
    print(crossover_point )
    
    #generate offsprings
    offspring1_genes = genes_parent1[:crossover_point] + genes_parent2[crossover_point:]
    offspring2_genes = genes_parent2[:crossover_point] + genes_parent1[crossover_point:]
    
    #generate dataframe for offsprings
    offspring1_df=pd.DataFrame()
    offspring2_df=pd.DataFrame()    
    day_pattern = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]
    offspring1_df['Day'] = day_pattern
    offspring2_df['Day'] = day_pattern    
    slot_pattern = ['1-4', '5-8', '9-12'] * 7  # Repeat this 3-slot pattern for 7 days
    offspring1_df['Slot'] = slot_pattern
    offspring2_df['Slot'] = slot_pattern
    
    for i in range(len(genes_parent1)):
        offspring1_df.loc[i, 'Opening Hour'] = offspring1_genes[i][0]
        offspring1_df.loc[i, 'Aircraft Number'] = offspring1_genes[i][1]

        offspring2_df.loc[i, 'Opening Hour'] = offspring2_genes[i][0]
        offspring2_df.loc[i, 'Aircraft Number'] = offspring2_genes[i][1]
    
    return offspring1_df,offspring2_df  

def dependent_variable_update_after_crossover(offsprings,arrivals_df_list,aircraft_df,max_waiting_time): 

    #Total arrival number list per day
    arrival_day=[]
    for i in arrivals_df_list:
        arrival_day.append(i['Passengers'].sum()) 
    
    # Define the time slots (0-4, 4-8, 8-12)
    time_slots = [(1, 4), (5, 8), (9, 12)]
    
    #Solution list to store solutions
    all_solutions = []
    
    for solution_num,offsprings in enumerate(offsprings):

        #Initializaing key variables in each solution
        carryover_df = pd.DataFrame(columns=['Hour', 'Passengers'])        
        total_flights = []        
        carryover_list = []
        spilled_list=[]
        total_day_list=[]    
        previous_carryover = 0
        previous_carryover_list=[] 
        day_total_waiting_time_list = []
        day_total_waiting_cost_list = []        
        track_combined_arrivals=[]
        
        # 7 günlük plan yapalım
        for day in range(7):
            
            #Reseting variables at the beginning of the each day
            carryover_day = 0  
            spilled_day=0
            total_assigned_day=0
            combined_arrivals=0
            day_total_waiting_time=0
            day_total_waiting_cost=0
            
            # Copy arrivals for the corresponding day
            arrivals_df = arrivals_df_list[day].copy()

            # Index adjustment if there is a carry over from previous day for carryovers and arrivals
            if not carryover_df.empty:
                carryover_df['Hour'] = carryover_df['Hour'] - 6
                rows_to_drop = carryover_df[carryover_df["Hour"] < 0].index
                carryover_df.drop(rows_to_drop, inplace=True)
                arrivals_df['Hour'] = arrivals_df['Hour'] + 6
                    
            # Merge previous day carryovers and arrivals for the day
            combined_arrivals = pd.concat([carryover_df, arrivals_df]).sort_values(by='Hour', ascending=True).reset_index(drop=True)
        
           
            for slot in time_slots:
                start_time, end_time = slot
                # Calling aircraft time and opening information from offspring
                aircraft_number = offsprings.loc[(offsprings['Day'] == day + 1) & (offsprings['Slot'] == f'{start_time}-{end_time}'), 'Aircraft Number'].values[0]
                opening_hour = offsprings.loc[(offsprings['Day'] == day + 1) & (offsprings['Slot'] == f'{start_time}-{end_time}'), 'Opening Hour'].values[0]
                aircraft_capacity = aircraft_df[aircraft_df['Aircraft Number'] == aircraft_number]['Capacity'].values[0]
               
                # Time adjustment
                adjusted_opening_hour = opening_hour
                actual_opening_hour=opening_hour
                
                #Adjut opening hour if there is a carryover from previous day
                if not carryover_df.empty:
                    adjusted_opening_hour = opening_hour + 6
                else:
                    adjusted_opening_hour = opening_hour
                                    
                # Reset 'waiting_time' and 'waiting_cost' columns to 0 at the beginning of each loop iteration
                combined_arrivals['waiting_time'] = 0
                combined_arrivals['waiting_cost'] = 0
                
                total_waiting_time=0
                total_waiting_cost=0
                        
                #Calculate waiting time for passenegrs in the combined arrivals (all arrivals from today+unassigned previous day)
                combined_arrivals["waiting_time"]=adjusted_opening_hour-combined_arrivals["Hour"]
                # Ensure that any negative waiting time is set to zero
                combined_arrivals['waiting_time'] = combined_arrivals['waiting_time'].apply(lambda x: x if x > 0 else 0)
                combined_arrivals["waiting_cost"]=combined_arrivals["waiting_time"].apply(calculate_waiting_cost)
                
                #Generate a set of passengers that are eligible till the aircraft opening and fit max waiting time criteria
                today_passengers = combined_arrivals[(combined_arrivals['Hour'] < adjusted_opening_hour) & (combined_arrivals['waiting_time'] < max_waiting_time)]
                #Find these eligible passngers sum
                total_passengers = today_passengers['Passengers'].sum()
                
                #if eligible passngers sum less or equal to the aircraft capacity, then assign all of them
                if total_passengers <= aircraft_capacity:
                    
                    #track assigned passengers
                    total=total_passengers
                    total_assigned_day += total_passengers
                    # Calculate total waiting time
                    a=combined_arrivals.copy()
                    total_waiting_time  = (a["waiting_time"] * a["Passengers"]).sum()
                    day_total_waiting_time= day_total_waiting_time + total_waiting_time 
                    #Calculate total waiting cost
                    total_waiting_cost = (a["waiting_cost"] * a["Passengers"]).sum()
                    day_total_waiting_cost=day_total_waiting_cost+ total_waiting_cost
                    
                    total_flights.append({
                        "Solution": solution_num + 1,
                        'Day': day + 1,
                        'Slot': f'{start_time}-{end_time}',
                        'Opening Hour': actual_opening_hour,
                        'Aircraft Capacity': aircraft_capacity,
                        "Aircraft Number":aircraft_number,
                        "Total passengers": arrivals_df['Passengers'].sum(),
                        'Assigned Passengers': total,
                        "Total waiting cost":total_waiting_cost,
                        "Total waiting time":total_waiting_time 
                    })
                    
                    #delete all passengers in the combined arrivals since they are all assigned
                    rows_to_drop = combined_arrivals[combined_arrivals['Hour'] < adjusted_opening_hour].index
                    combined_arrivals.drop(rows_to_drop, inplace=True)

                    
                else:
                    #Generate a set of passengers that are eligible till the aircraft opening and fit max waiting time criteria
                    remaining_passengers = combined_arrivals[(combined_arrivals['Hour'] < adjusted_opening_hour) & (combined_arrivals['waiting_time'] < max_waiting_time)]
                    #Sort them based on arrival time in ascending order(gelişi ilk olan ilk değerlendirilcek)
                    remaining_passengers_sorted = remaining_passengers.sort_values(by='Hour')
                    
                    #Assigned passenger list under this else
                    passengers_to_board = []  #assigned passenger list
                    total = 0 # initialize zero to track total assigned
                    
                    #for waiting time and cost calculation
                    total_waiting_time=0
                    total_waiting_cost=0
                    
                    combined_arrivals.reset_index(drop=True, inplace=True)
                    
                    # Initialize list to store conditions to delete from combined_arrivals
                    rows_to_delete_from_combined = []
                    
                    #copy combined_arrivals
                    copy_combined_arrivals_df=combined_arrivals.copy()
        
                    for index, row in remaining_passengers_sorted.iterrows():
                        
                        if aircraft_capacity >= row['Passengers']:
                            # Track the row in combined_arrivals by matching 'Hour' and 'Passengers' and delete them
                            condition = (combined_arrivals['Hour'] == row['Hour']) & (combined_arrivals['Passengers'] == row['Passengers'])
                            matched_row_index = combined_arrivals[condition].index
                            if not matched_row_index.empty:
                                #delete from combined_df
                                rows_to_delete_from_combined.append(matched_row_index[0])
                                #calculate waiting cost and time
                                total_waiting_time=(copy_combined_arrivals_df.loc[matched_row_index, 'waiting_time']*copy_combined_arrivals_df.loc[matched_row_index, 'Passengers']).sum()                        
                                total_waiting_cost=(copy_combined_arrivals_df.loc[matched_row_index, 'waiting_cost']*copy_combined_arrivals_df.loc[matched_row_index, 'Passengers']).sum()
                            #update reamining paasengers and aircraft capacity
                            passengers_to_board.append(row)
                            aircraft_capacity -= row['Passengers']
                            total += row['Passengers']
                            remaining_passengers_sorted.drop(index, inplace=True)
        
                            
                            
                        elif aircraft_capacity > 0:
                            
                            # Adjust row for partial assignment in combined_arrivals
                            condition = (combined_arrivals['Hour'] == row['Hour']) & (combined_arrivals['Passengers'] == row['Passengers'])
                            matched_index = combined_arrivals[condition].index[0]
                            remaining_capacity = row['Passengers'] - aircraft_capacity
                            combined_arrivals.at[matched_index, 'Passengers'] = remaining_capacity
                            
                            #calculate waiting cost and waiting time
                            total_waiting_time=(copy_combined_arrivals_df.loc[matched_index,'waiting_time']*copy_combined_arrivals_df.loc[matched_index,'Passengers']).sum() 
                            total_waiting_cost=(copy_combined_arrivals_df.loc[matched_index,'waiting_cost']*copy_combined_arrivals_df.loc[matched_index,'Passengers']).sum() 
                            
                            #Partial assignment
                            row['Passengers'] = row['Passengers'] - aircraft_capacity
                            passengers_to_board.append(aircraft_capacity)
                            total += aircraft_capacity
                            aircraft_capacity = 0
                            remaining_passengers_sorted.at[index, 'Passengers'] -= aircraft_capacity
                            
                        else:  # capacity is full
                            aircraft_capacity = 0
                            break
                    
                    # Drop the rows from combined_arrivals for the all passengers in same interval assigned case(first if)
                    combined_arrivals.drop(rows_to_delete_from_combined, inplace=True)

                    # Add to total boarded passengers for the day (assigned)
                    total_assigned_day += total  
                    
                    #Add to totaltotal waiting cost and time for the day 
                    day_total_waiting_time += total_waiting_time
                    day_total_waiting_cost += total_waiting_cost                    
                                        
                    # Uçağa binen yolcuları ekliyoruz
                    if passengers_to_board:
                        total_flights.append({
                            "Solution": solution_num + 1,
                            'Day': day + 1,
                            'Slot': f'{start_time}-{end_time}',
                            'Opening Hour': actual_opening_hour,
                            'Aircraft Capacity': aircraft_df[aircraft_df['Aircraft Number'] == aircraft_number]['Capacity'].values[0],
                            "Aircraft Number":aircraft_number,
                            "Total passengers": arrivals_df['Passengers'].sum(),
                            'Assigned Passengers': total,  # total assigned, 
                            "Total waiting cost":total_waiting_cost,
                            "Total waiting time":total_waiting_time 
                        })
                        
                      
            #used to calculate spilled and carryover in a day for each time slot
            track_combined_arrivals.append(combined_arrivals)
            
            #Tracking carryovers
            #if there are carryovers from previous day
            if not carryover_df.empty:
            # Candidates for carryover to next day if combined_arrivals has hours greater than 12
                carryover_candidates = combined_arrivals[combined_arrivals['Hour'] >=12]
            else:
            # Candidates for carryover from combined arrivals if hours are greater than 6
                carryover_candidates = combined_arrivals[combined_arrivals['Hour'] >=6]
            # Reset carryover_df to include new carryover candidates, and handle empty case gracefully
            if not carryover_candidates.empty:
                carryover_df = carryover_candidates.reset_index(drop=True)
            else:
                carryover_df = pd.DataFrame(columns=['Hour', 'Passengers',"waiting_time","waiting_cost"])  # Reset to an empty DataFrame with correct columns        
            # Sum up carryover passengers for the day
            carryover_day = carryover_df["Passengers"].sum()        
            # At the end of the day, append the total carryover passengers
            carryover_list.append(carryover_day)
            
            #Calculate total assigned for each day
            total_day_list.append(total_assigned_day)
            
            #Calculate total waiting time and cost for day
            day_total_waiting_time_list.append(day_total_waiting_time)
            day_total_waiting_cost_list.append(day_total_waiting_cost)
            
            if day == 0:
                previous_carryover=0
            else:
                previous_carryover=carryover_list[day-1]
            previous_carryover_list.append(previous_carryover)   
                 
            #At the end of the day, append the spilled passengers
            spilled_day= max(arrivals_df['Passengers'].sum() + previous_carryover - total_assigned_day - carryover_df["Passengers"].sum(), 0)
            spilled_list.append(spilled_day)
        """                           
        summary_df=pd.DataFrame({
            "Day":list(range(1,8)),
            "Total Passengers":arrival_day,
            "Carryover Previous Day":previous_carryover_list,
            "Assigned":total_day_list,
            "Carryover Today":carryover_list,
            "Spilled":spilled_list,
            "Day total waiting cost":day_total_waiting_cost_list,
            "Day total waiting time":day_total_waiting_time_list,
            
            })
        """
        # Write total flights results to a dataframe
        final_schedule_df = pd.DataFrame(total_flights)
               
        #Find carryover and spilled for each slot and day        
        results_list = []
        
        for index, i in enumerate(track_combined_arrivals):
            
            if (i['Hour'] > 12).any():    
            
            # Process the first slot (1-10 hours)
                spilled_first_phase = i[(i['Hour'] >= 1) & (i['Hour'] < 10)]['Passengers'].sum()
            
            # Process the second slot (10-12 hours)
                spilled_second_phase = i[(i['Hour'] >= 10) & (i['Hour'] < 12)]['Passengers'].sum()
        
            # Process the third slot (greater than 12 hours, considered spilled)
                spilled_third_phase = 0
                
            # Process the third slot (greater than 12 hours, considered spilled)
                carried_first_phase = 0
                
            #carried passengers
                carried_second_phase = i[(i['Hour'] >= 12) & (i['Hour'] <= 14)]['Passengers'].sum()
            #carried passengers
                carried_third_phase = i[(i['Hour'] > 14) & (i['Hour'] <= 18)]['Passengers'].sum()
                
            else:
                
                # Process the first slot (1-10 hours)
                spilled_first_phase = i[(i['Hour'] >= 1) & (i['Hour'] < 5)]['Passengers'].sum()
            
            # Process the second slot (10-12 hours)
                spilled_second_phase = i[(i['Hour'] == 5)]['Passengers'].sum()
        
            # Process the third slot (greater than 12 hours, considered spilled)
                spilled_third_phase = 0
                
            # Process the third slot (greater than 12 hours, considered spilled)
                carried_first_phase = 0
                
            #carried passengers
                carried_second_phase = i[(i['Hour'] >= 6) & (i['Hour'] <= 8)]['Passengers'].sum()
            #carried passengers
                carried_third_phase = i[(i['Hour'] > 8) & (i['Hour'] <= 12)]['Passengers'].sum()
        
            # Append results to the list as a dictionary
            results_list.append({
                "Day": index+1,
                "Slot": "1-4",
                "Carryover": carried_first_phase if not pd.isna(carried_first_phase) else 0,
                "Spilled": spilled_first_phase if not pd.isna(spilled_first_phase) else 0
            })
        
            results_list.append({
                "Day": index+1,
                "Slot": "5-8",
                "Carryover": carried_second_phase if not pd.isna(carried_second_phase) else 0,
                "Spilled": spilled_second_phase if not pd.isna(spilled_second_phase) else 0
            })
        
            results_list.append({
                "Day": index+1,
                "Slot": "9-12",
                "Carryover":carried_third_phase if not pd.isna(carried_third_phase) else 0,
                "Spilled": spilled_third_phase if not pd.isna(spilled_third_phase) else 0
            })
        
        # Write results_list to a DataFrame
        final_results_df = pd.DataFrame(results_list)
       
        # Merge final_results_df and final_schedule_df based on the columns 'Day' and 'Slot'
        chromosome_df = pd.merge(final_schedule_df, final_results_df, on=['Day', 'Slot'], how='left')
                
        all_solutions.append(chromosome_df)
                        
    return all_solutions
    
def mutation_assigned_passengers(offsprings, aircraft_df):
    # Define the time slots with their corresponding ranges
    time_slots = [(1, 4), (5, 8), (9, 12)]
    
    for offspring_df in offsprings:
        # Iterate through each row in the dataframe
        for idx, row in offspring_df.iterrows():
            # Check if the assigned passengers for this row are zero
            if row['Assigned Passengers'] == 0:
                # Determine which time slot the current row belongs to
                current_opening_hour = row['Opening Hour']
                for start_time, end_time in time_slots:
                    if start_time <= current_opening_hour <= end_time:
                        # Randomly select a new aircraft type
                        selected_aircraft = aircraft_df.sample(1).iloc[0]
                        new_aircraft_number = selected_aircraft['Aircraft Number']
                        new_aircraft_capacity = selected_aircraft['Capacity']

                        # Randomly select a new opening hour within the identified slot range
                        new_opening_hour = np.random.randint(start_time, end_time + 1)

                        # Update the dataframe with the new aircraft type and opening hour
                        offspring_df.at[idx, 'Aircraft Number'] = new_aircraft_number
                        offspring_df.at[idx, 'Aircraft Capacity'] = new_aircraft_capacity
                        offspring_df.at[idx, 'Opening Hour'] = new_opening_hour
                        print("Solution number ",offspring_df.at[idx, 'Solution'],"Index ",idx,"Aircraft number ",new_aircraft_number,"Aircraft cap ",new_aircraft_capacity,"Opening hour ",new_opening_hour)
                        break  # Stop checking slots once a match is found

    return offsprings
    
def elitist_replacement(initial_solutions,fitness_initial_solutions, all_offsprings_after_mutation, num_initial_population):
    # Step 1: Calculate fitness value for all offsprings
    fitness_offsprings = fitness_function(all_offsprings_after_mutation, 1000, 2000, 3000, 500, 300)
    
    # Step 2: Combine initial solutions and offsprings fitness DataFrames
    combined_fitness = pd.concat([fitness_initial_solutions, fitness_offsprings], ignore_index=True).reset_index(drop=True)
    
    # Step 3: Combine the actual DataFrames from initial_solutions and all_offsprings_after_mutation into a single list
    combined_solutions = initial_solutions + all_offsprings_after_mutation
    
    # Step 4: Reset the index of combined_solutions for alignment with combined_fitness
    combined_solutions = [df.reset_index(drop=True) for df in combined_solutions]
    
    combined_solutions_copy =copy.deepcopy(combined_solutions)

    # Step 5: Sort the combined fitness DataFrame by fitness score in descending order
    sorted_indices = combined_fitness.sort_values(by='Fitness Score', ascending=False).index[:num_initial_population]
    
    # Step 6: Select the corresponding DataFrames based on sorted fitness indices
    selected_solutions = [combined_solutions[i] for i in sorted_indices]
    
    # Step 7: Assign new solution numbers to each selected DataFrame
    for solution_num, solution_df in enumerate(selected_solutions, 1):
        solution_df['Solution'] = solution_num
        
    return combined_fitness,combined_solutions_copy ,sorted_indices,selected_solutions
    
def genetic_algorithm(num_initial_population,num_children):    
    initial_solutions=generate_initial_solutions(num_initial_population,arrivals_df_list,aircraft_df,8)
    fitness_initial_solutions=fitness_function(initial_solutions,1000,2000,3000,500,300)
    fitness_proportional_selection=proportional_probability_selection(fitness_initial_solutions)
    
    #Generate a list to store all offsprings
    all_offsprings_before_mutation=[]
    all_offsprings_after_mutation=[]
    all_offsprings_with_parents=[]
    
    for i in range(int(num_children/2)):
        
        #select parents using roulette wheel selection approach
        parents=roulette_wheel_selection(fitness_proportional_selection)
        parent1=parents[0]
        parent2=parents[1]
        
        #Crossover single point method and generate two children
        offsprings=crossover_single_point(parent1,parent2,initial_solutions)
       
        #update depended variables
        offsprings_chromosome=dependent_variable_update_after_crossover(offsprings,arrivals_df_list,aircraft_df,8)        
        offsprings_chromosome_before_mutation=copy.deepcopy(offsprings_chromosome)
        all_offsprings_before_mutation.extend(offsprings_chromosome_before_mutation)
        
        #mutation based on unassigned
        mutated_offsprings=mutation_assigned_passengers(offsprings_chromosome, aircraft_df)
        offsprings_chromosome_after_mutation=dependent_variable_update_after_crossover(mutated_offsprings,arrivals_df_list,aircraft_df,8)
        
        #Store offspring results
        all_offsprings_after_mutation.extend(offsprings_chromosome_after_mutation)
        

        for idx, offspring_df in enumerate(offsprings_chromosome_after_mutation):
        # Record the parents for each offspring
            parent_info = (parent1, parent2)
            all_offsprings_with_parents.append((parent_info, offspring_df))
            
        parents=None
        parent1=None
        parent2=None
        
    selected_solutions=elitist_replacement(initial_solutions,fitness_initial_solutions, all_offsprings_after_mutation, num_initial_population)
    
    return initial_solutions,all_offsprings_before_mutation,all_offsprings_after_mutation,all_offsprings_with_parents,selected_solutions

aaa=genetic_algorithm(20,10) 