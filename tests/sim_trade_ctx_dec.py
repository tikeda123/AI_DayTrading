
import pandas as pd
import os,sys


from trading_analysis_kit.trading_context import initialize_components,TradingContext,IdleState,EntryPreparationState,PositionState
from common.trading_logger import TradingLogger
from common.config_manager import ConfigManager

#from fxtransaction import FXTransaction
#from aiml.inference_prediction_manager import InferencePredictionManager
class SimulationEntryPreparationStateDecorator(EntryPreparationState):
    def __init__(self, decorated_context):
        self._decorated_context = decorated_context

    def event_handle(self, context, index: int):
        if context.decorated_context.get_entry_counter() >= 2:
            context.chage_state(SimulationIdleStateDecorator(IdleState()))
            print(context.get_state_classname())

        self._decorated_context.event_handle(context.decorated_context, index)



class SimulationIdleStateDecorator(IdleState):
    def __init__(self, decorated_context):
        self._decorated_context = decorated_context

    def event_handle(self, context, index: int):
        self._decorated_context.event_handle(context.decorated_context, index)

        if context.get_state_classname() == 'EntryPreparationState':
            context.chage_state(SimulationEntryPreparationStateDecorator(EntryPreparationState()))
            print(context.get_state_classname())

class SimulationTradingContextDecorator(TradingContext):
    #def __init__(self, decorated_context, fxt: FXTransaction, prediction_manager: InferencePredictionManager):
    def __init__(self, decorated_context):
        self.state = SimulationIdleStateDecorator(IdleState())
        self.decorated_context = decorated_context
        self.__config_manager = ConfigManager.get_instance()
        self.__logger = TradingLogger.get_instance()
        #self.__fxt = fxt
        #self.__prediction_manager = prediction_manager()

    def event_handle(self,index:int):
        self.set_current_index(index)
        self.state.event_handle(self,index)

    def get_state_classname(self):
        return self.decorated_context.get_state_classname()
    
    def chage_state(self, state):
        self.state = state

def main():
    config_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/aitrading_settings_ver2.json'
    # Initialization
    config_manager, trading_logger, dataloader, trading_analysis = initialize_components(config_path)

    sim_trading_ctx = SimulationTradingContextDecorator(TradingContext())


    # Iterate over the analyzed data and generate events based on the data
    for index in range(30):
       sim_trading_ctx.event_handle(index)

if __name__ == '__main__':
    main()




    






