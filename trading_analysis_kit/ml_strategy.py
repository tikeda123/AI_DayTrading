import pandas as pd
import os,sys

sys.path.append('/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0')

from trading_analysis_kit.trading_state import *
from trading_analysis_kit.trading_strategy import TradingStrategy
from trading_analysis_kit.simulation_strategy_context import SimulationStrategyContext

class MLDataCreationStrategy(TradingStrategy):
    def Idel_event_execute(self, context : SimulationStrategyContext):
        context.dm.record_state(STATE_IDLE)
        context.change_to_entrypreparation_state()

    def EntryPreparation_event_execute(self, context: SimulationStrategyContext):
        context.dm.set_entry_index(context.dm.get_current_index())
        context.dm.set_entry_price(context.dm.get_close_price())

        bb_direction = context.dm.get_bb_direction()

        if bb_direction == BB_DIRECTION_UPPER:
            context.dm.set_prediction(PRED_TYPE_LONG)
        elif bb_direction == BB_DIRECTION_LOWER:
            context.dm.set_prediction(PRED_TYPE_LONG)

        context.dm.record_state(STATE_ENTRY_PREPARATION)
        context.change_to_position_state()

    def PositionState_event_exit_execute(self, context: SimulationStrategyContext):
        context.dm.record_state(STATE_POSITION)
        context.set_current_max_min_pandl()

        context.dm.set_exit_index(context.dm.get_current_index())
        context.dm.set_exit_price(context.dm.get_close_price())

        profit = context.calculate_current_profit()
        context.dm.set_bb_profit(profit, context.dm.get_entry_index())
        context.record_entry_exit_price()
        context.record_max_min_pandl_to_entrypoint()
        context.change_to_idle_state()

    def PositionState_event_continue_execute(self, context : SimulationStrategyContext):
        context.dm.record_state(STATE_POSITION)
        context.set_current_max_min_pandl()

        current_profit = context.calculate_current_profit()
        context.dm.set_current_profit(current_profit)

    def decide_on_position_exit(self, context: SimulationStrategyContext, index:int):
        bb_direction = context.dm.get_bb_direction()
        pred = context.dm.get_prediction()

        position_state_dict = {
            (BB_DIRECTION_UPPER, PRED_TYPE_LONG): ['less_than', COLUMN_MIDDLE_BAND],
            (BB_DIRECTION_UPPER, PRED_TYPE_SHORT): ['less_than', COLUMN_MIDDLE_BAND],
            (BB_DIRECTION_LOWER, PRED_TYPE_LONG): ['greater_than', COLUMN_MIDDLE_BAND],
            (BB_DIRECTION_LOWER, PRED_TYPE_SHORT): ['greater_than', COLUMN_MIDDLE_BAND]
        }

        condition = position_state_dict.get((bb_direction, pred))
        if condition is None:
            return 'PositionState_event_continue_execute'

        operator, column = condition

        if operator == 'less_than':
            if context.is_first_column_less_than_second(COLUMN_CLOSE, column,index):
                return 'PositionState_event_exit_execute'
        elif operator == 'greater_than':
            if context.is_first_column_greater_than_second(COLUMN_CLOSE, column,index):
                return 'PositionState_event_exit_execute'

        return 'PositionState_event_continue_execute'