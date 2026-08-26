from agentfem_learning.neural_fields.xdem.extension import extension


def test_extension_identity_and_capabilities_are_explicit():
    assert extension.spec.name == "agentfem-learning.xdem"
    assert extension.spec.version == "0.1.0a1"
    assert "learning.neural_field.energy" in extension.spec.capabilities
    assert "fracture.williams_mode_iii_reference" in extension.spec.capabilities
    assert "fracture.williams_vector_reference" in extension.spec.capabilities
    assert "fracture.stress_intensity_extraction" in extension.spec.capabilities
