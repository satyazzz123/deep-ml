def conditional_probability(joint_distribution: dict) -> float:

	
    P_B=joint_distribution[('A','B')]+joint_distribution[('`A','B')]

    return joint_distribution[('A','B')]/P_B