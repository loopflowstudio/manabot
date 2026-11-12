import managym

layout = managym.observation_layout()
col = layout["agent_permanents"]["columns"]["definition"]
index: int = col["index"]
kind: str = col["kind"]
shape = layout["agent_permanents"]["shape"]
row_kinds = layout["facts"]["row_kinds"]
tokens: managym.CatalogTokens = managym.catalog_tokens()
old = managym.learning_catalog()
bad: str = col["index"]
