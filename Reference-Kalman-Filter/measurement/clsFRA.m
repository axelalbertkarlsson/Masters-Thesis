% Class for Forward Rate Agreements (FRA)

classdef clsFRA < handle
   properties
      assetType = 'FRA'
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
      
      function fra = clsFRA(ric, fraID)
        fra.assetRIC = ric;
        [startDate, endDate, timeFrac] = mexPortfolio('cashFlowsFRA', fraID);
        fra.valueDate = startDate;
        fra.maturityDate = endDate;
        fra.dt = timeFrac;
      end % clsInstruments constructor
      
      function lastDate = get.lastDate(fra)
        lastDate = fra.maturityDate;
      end % Function is called when value is needed
      
      function set.lastDate(fra,~)
         error('You cannot set lastDate explicitly');
      end
      
      function cfDates = getCashFlowDates(fra)
        cfDates = [fra.valueDate ; fra.maturityDate];
      end
      
   end
   
   methods (Access = 'private') % Access by class members only
   end
end % classdef
