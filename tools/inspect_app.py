import inspect
import pydantic
import lucia_ultimate_quantum_integrated_fixed as mod

print("MODULE_FILE:", getattr(mod, "__file__", "??"))
print("PYDANTIC:", getattr(pydantic, "__version__", repr(pydantic)))
print("\nAI_CHAT_SOURCE:\n")
print(inspect.getsource(mod.ai_chat))
