# Household chore optimizer
Tool for scheduling household chores, optimizing for the most equal distribution of workload among household members and over time.

## Optimization problem formulation
Consider the following constraints:
1. Each household chore must be scheduled at regular intervals, where the length of the intervals depends on the chore.
2. Each chore must be scheduled the number of times given by dividing the total number of periods by the interval of the chore (floored).
3. The household chores scheduled for a given period must each be assigned to a household member.
4. Each household chore must be assigned approximately the same number of times to each household member.

Let  
$X_{mcp} = 1$ if household member $m$ is assigned to chore $c$ during period $p$  
$I_c =$ the required time interval in number of periods between each time chore $c$ is scheduled  
$N =$ the total number of periods  
$M =$ the total number of household members  
$T_c =$ the time required to complete chore $c$  

Then the preceding constraints can be modelled as
1. $\forall_c\forall_{p \in [0,N - I_c]} \sum_m X_{mc,p} = \sum_m X_{mc,p+I_c}$
2. $\forall_c\sum_m\sum_p X_{mcp} = \left\lfloor \dfrac{N}{I_c} \right\rfloor$
3. $\forall_c\forall_p \sum_m X_{mcp} \leq 1$
4. $\forall_c \forall_m \sum_p X_{mcp} \leq \left\lceil \dfrac{N}{I_c M} \right\rceil$

The goal is to evenly distribute the chores across both the periods and among household members. Hence we seek to both minimize the maximum workload of any one household member in any period, and to minimize the difference between total workload assigned to each family member across all periods.

Let  
$w_a =$ the weight assigned to the objective of minimizing maximum workload in a period  
$w_b =$ the weight assigned to the objective of minimizing the difference between household members

Then the objective function is to  
$$minimize \quad \max\forall_m\forall_p w_a \sum_c T_c X_{mcp} + \max \forall_{i \in m} \forall_{j \in m} w_b \left\lvert \sum_c \sum_p T_c X_{m_icp} - \sum_c \sum_p T_c X_{m_jcp} \right\rvert$$


## Sources
Fred Glover and Claude McMillan.
[The General Employee Scheduling Problem: An Integration of MS and AI](https://leeds-faculty.colorado.edu/glover/fred%20pubs/171%20-%20General%20Employee%20Scheduling%20Problem%2086%20TS.pdf).
Computers & Operations Research 13(5):563-573, January 1986.
