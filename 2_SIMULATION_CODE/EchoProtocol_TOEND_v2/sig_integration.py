# sig_integration.py
def handle_final_states(sig_graph):
    final_nodes = sig_graph.query("""
        MATCH (n:Identity) 
        WHERE n.final_state IS NOT NULL
        RETURN n
    """)
    
def integrate_guardians(sig, guardian):  
    if guardian.identity.λ > 2.0:  
        sig.adjust_edge("narrative_risk", weight=0.0)  # Désactive les bords risqués  
    
    for node in final_nodes:
        if node['state_type'] == 'SINGULARITY':
            sig_graph.create(
                f"CREATE (s:Singularity {{id: '{node.id}', timestamp: '{node.timestamp}'})"
                f"MERGE (n)-[r:EVOLVED_TO]->(s)"
            )