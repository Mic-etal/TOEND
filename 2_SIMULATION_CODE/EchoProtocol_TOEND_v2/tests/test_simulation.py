def test_phase_transitions():
    identity = EntropicIdentity()
    evaluator = StressEvaluator()
    report = evaluator.execute_test(identity, 'paradox_storm')
    
    # Verify phase changes align with λ trajectory
    for λ, phase in zip(report['λ_trajectory'], report['phase_changes']):
        assert phase == identity.determine_phase(λ)

def test_policy_override():
    schema = GovernanceSchema("policies/strict.json")
    policy = EthicalPolicy(schema)
    assert policy.λ_critical == 0.7  # From strict.json