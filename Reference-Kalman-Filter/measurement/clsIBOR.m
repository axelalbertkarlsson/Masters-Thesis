% Class for Interbank offered rate (IBOR)

classdef clsIBOR < handle
   properties
      assetType = 'IBOR'
      assetRIC = ''
      
      settlementDate = 0    % The date in which the payment occurs
      maturityDate = 0
      tradeDate = 0         % The current date for which porperties are set
      price = 0             % Price noted on the trade day
      
      dt = 0
      
      active = true         % Use this contract for building the curve
   end
   properties (Dependent) % Values that are calculated
      lastDate
   end
   
   methods
      
      function ibor = clsIBOR(ric, depositID)
        ibor.assetRIC = ric;
        [tradeDate, settlementDate, maturityDate, timeFrac] = mexPortfolio('cashFlowsDeposit', depositID);
        ibor.tradeDate = tradeDate;
        ibor.settlementDate = settlementDate;
        ibor.maturityDate = maturityDate;
        ibor.dt = timeFrac;        
      end % clsInstruments constructor
           
      function lastDate = get.lastDate(ibor)
        lastDate = ibor.maturityDate;
      end % Function is called when value is needed
      
      function set.lastDate(ibor,~)
         error('You cannot set lastDate explicitly');
      end
      
      function cfDates = getCashFlowDates(ibor)
        cfDates = [ibor.settlementDate; ibor.maturityDate];
      end
      
   end
   
   methods (Access = 'private') % Access by class members only
   end
end % classdef
