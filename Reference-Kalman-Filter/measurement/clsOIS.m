% Class for Overnight Indexed Swap (OIS)

classdef clsOIS < handle
   properties
      assetType = 'OIS'
      assetRIC = ''
      
      settlementDate = 0    % The date in which the payment occurs
      maturityDate = 0
      tradeDate = 0         % The current date for which porperties are set
      price = 0             % Price noted on the trade day
      
      nFix = 0              % Number of cashflows in the fixed leg
      cfDatesFix = 0        % Dates for casflows in the fixed leg
      dtFix = 0
      
      nFlt = 0              % Number of cashflows in the floating leg
      cfDatesFlt = 0        % Dates for casflows in the floating leg
      dtFlt = 0
      
      active = true         % Use this contract for building the curve
   end
   properties (Dependent) % Values that are calculated
      lastDate
   end
   
   methods
         
      function ois = clsOIS(ric, oisID)
        [tradeDate, settlementDate, maturityDate, floatCouponFixingTimes, floatCouponStartDates, floatQuasiDates, floatCouponDates, floatTimeFrac, fixedQuasiDates, fixedCouponDates, fixedTimeFrac] = mexPortfolio('cashFlowsOIS', oisID);
        ois.tradeDate = tradeDate;
        ois.maturityDate = maturityDate;
        ois.settlementDate = settlementDate;
        ois.cfDatesFix = fixedCouponDates;
        ois.cfDatesFlt = floatCouponDates;
        ois.dtFix = fixedTimeFrac;
        ois.dtFlt = floatTimeFrac;
        ois.assetRIC = ric;

        ois.nFix = length(ois.cfDatesFix);
        ois.nFlt = length(ois.cfDatesFlt);
        
        if (ois.cfDatesFix(end) ~= ois.maturityDate)
          error('This is assumed to determine the last cash flow date');
        end
      end % clsInstruments constructor
             
      function lastDate = get.lastDate(ois)
        lastDate = ois.maturityDate;
      end % Function is called when value is needed
      
      function set.lastDate(ois,~)
         error('You cannot set lastDate explicitly');
      end
      
      function cfDates = getCashFlowDates(ois)
        cfDates = sort([ois.cfDatesFix; ois.cfDatesFlt]);
      end
      
   end
   
   methods (Access = 'private') % Access by class members only
   end
end % classdef
