% Class for Future - Interest Rate

classdef clsFutureIR < handle
   properties
      assetType = 'futureIR'
      assetRIC = ''
      
      valueDate = 0         % The date in which the payment occurs
      maturityDate = 0
      tradeDate = 0         % The current date for which porperties are set
      price = 0             % Price noted on the trade day
      
      dt = 0
      
      active = true         % Use this contract for building the curve
      
      
      % properties needed for Tenor curve methods
      dValueDate = 0
      dMaturityDate = 0
   end
   properties (Dependent) % Values that are calculated
      lastDate
   end
   
   methods
      
      function future = clsFutureIR(ric, futureID)
        future.assetRIC = ric;

        [maturityDate, couponStartDate, couponEndDate, timeFrac] = mexPortfolio('cashFlowsFutureIR', futureID);
        future.valueDate = couponStartDate;
        future.maturityDate = couponEndDate;
        future.dt = timeFrac;     
      end % clsInstruments constructor
      
      function lastDate = get.lastDate(future)
        lastDate = future.maturityDate;
      end % Function is called when value is needed
      
      function set.lastDate(future,~)
         error('You cannot set lastDate explicitly');
      end
      
      function cfDates = getCashFlowDates(future)
        cfDates = [future.valueDate ; future.maturityDate];
      end
      
   end
   
   methods (Access = 'private') % Access by class members only
   end
end % classdef
