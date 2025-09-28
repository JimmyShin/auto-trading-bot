from auto_trading_bot.alerts import start_alert_scheduler

def guard(x):
    print('GUARD_SWITCH_TO_TESTNET', x)

def stub(msg, **k):
    print('SLACK:', msg)
    return True

sch = start_alert_scheduler({'heartbeat_interval_sec': 0}, metrics_manager=None,
                            slack_sender=stub, interval_sec=0.1, guard_action=guard)
print('started')
for _ in range(2):
    sch.run_step()
print('stopped')
