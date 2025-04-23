% Class for Future - Interest Rate

classdef clsFutureCommodity < handle
   properties
      assetType = 'futureCommodity'
      assetRIC = ''
      
      tradeDate = 0
      maturityDate = 0
      price = 0             % Price noted on the trade day
      
      active = true         % Use this contract for building the curve
      
      
      % properties needed for Tenor curve methods
      dValueDate = 0
      dMaturityDate = 0
   end
   properties (Dependent) % Values that are calculated
      lastDate
   end
   
   methods
      
      function futureCommodity = clsFutureCommodity(ric, futureID, tradeDate)
        futureCommodity.assetRIC = ric;

        maturityDate = mexPortfolio('maturityDate', futureID, tradeDate);
        futureCommodity.maturityDate = maturityDate;
      end % clsInstruments constructor
      
      function lastDate = get.lastDate(future)
        lastDate = future.maturityDate;
      end % Function is called when value is needed
      
      function set.lastDate(futureCommodity,~)
         error('You cannot set lastDate explicitly');
      end
      
      function cfDates = getCashFlowDates(future)
        cfDates = [future.tradeDate ; future.maturityDate];
      end
      
   end
   
   methods (Access = 'private') % Access by class members only
   end
end % classdef
