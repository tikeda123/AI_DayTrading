import pandas as pd
import os,sys

sys.path.append('/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0')

from trading_analysis_kit.trading_state import *
from trading_analysis_kit.trading_strategy import TradingStrategy
from trading_analysis_kit.simulation_strategy_context import SimulationStrategyContext

class MLDataCreationStrategy(TradingStrategy):
    def Idel_event_execute(self, context : SimulationStrategyContext):
        context.dataloader.record_state(STATE_IDLE)
        context.change_to_entrypreparation_state()

    def EntryPreparation_event_execute(self, context: SimulationStrategyContext):
        context.dataloader.record_state(STATE_ENTRY_PREPARATION)
        context.change_to_position_state()

    def PositionState_event_exit_execute(self, context: SimulationStrategyContext):
        context.dataloader.record_state(STATE_POSITION)
        context.dataloader.ts.exit_index = context.dataloader.get_current_index()
        context.dataloader.set_exit_price(context.dataloader.get_close_price())

        profit = context.calculate_current_profit()
        context.dataloader.set_bb_profit(profit, context.dataloader.ts.entry_index)
        context.record_entry_exit_price()
        context.change_to_idle_state()

    def PositionState_event_continue_execute(self, context : SimulationStrategyContext):
        context.dataloader.record_state(STATE_POSITION)
        context.calculate_current_profit()

    def decide_on_position_exit(self, context: SimulationStrategyContext, index:int):
        bb_direction = context.dataloader.get_bb_direction()
        if bb_direction == BB_DIRECTION_UPPER and \
                context.is_first_column_less_than_second(index, COLUMN_CLOSE, COLUMN_MIDDLE_BAND):
            return 'PositionState_event_exit_execute'
        elif bb_direction == BB_DIRECTION_LOWER and \
                context.is_first_column_greater_than_second(index, COLUMN_CLOSE, COLUMN_MIDDLE_BAND):
            return 'PositionState_event_exit_execute'
        else:
            return 'PositionState_event_continue_execute'