# GeneticAlgorithm-AirlineScheduling

This repository provides the implementation of a Genetic Algorithm (GA) designed specifically for new route outbound scheduling in a mega-hub airline network. The model focuses on constructing efficient outbound schedules for newly introduced international routes, where no historical timetable exists and passenger demand must be matched with feasible aircraft operations.

The GA optimizes multiple components simultaneously, including outbound departure times, aircraft type assignment, aircraft availability, turnaround times, capacity limits, and allowable passenger waiting times. Each chromosome represents a full outbound-day schedule, and the GA iteratively improves solutions through selection, crossover, mutation, and elitist replacement.
This approach enables airlines to test and generate competitive outbound schedules for newly opened routes while balancing profitability, operational feasibility, and service quality.

Related Publication
This GA methodology is based on my peer-reviewed article:
Tacoglu, M., Ornek, M. A., & Kazancoglu, Y. (2025).
Genetic Algorithm and Mathematical Modelling for Integrated Schedule Design and Fleet Assignment in Mega-Hub Networks.
Aerospace, 12(6), 545.
🔗 https://www.mdpi.com/2226-4310/12/6/545

If you use this genetic algorithm or the code provided here, please reference this publication.
