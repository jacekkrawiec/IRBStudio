from irbstudio.config.schema import Config, Scenario

print('import OK')

try:
    Config()
    print('Config() did NOT raise')
except Exception as e:
    print('Config() raised:', type(e).__name__, e)

c = Config(scenarios=[Scenario(name='base', pd_auc=0.75)])
print('Created config with scenario:', c.scenarios[0].name, c.scenarios[0].pd_auc)
