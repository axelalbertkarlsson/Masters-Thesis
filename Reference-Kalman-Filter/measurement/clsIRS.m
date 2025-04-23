% Class for Interest Rate Swap (IRS)

classdef clsIRS < handle
   properties
      assetType = 'IRS'
      assetRIC = ''
      
      settlementDate = 0 
      maturityDate = 0
      tradeDate = 0         % The current date for which porperties are set
      price = 0             % Price noted on the trade day
      
      nFix = 0              % Number of cashflows in the fixed leg
      cfDatesFix = 0        % Dates for casflows in the fixed leg
      dtFix = 0
      
      nFlt = 0              % Number of cashflows in the floating leg
      cfDatesFlt = 0        % Dates for casflows in the floating leg
      dtFlt = 0
      cfDatesIborFlt = 0    % Dates when ibor rate ends
      dtIborFlt = 0         % Time for Ibor rate
      fixingDatesFlt = 0
      startDatesFlt = 0
      
      active = true         % Use this contract for building the curve
      
      % properties needed for Tenor curve methods
      dSettlementDate = 0
      dFix = 0
      dFlt = 0
   end
   
   properties (Dependent) % Values that are calculated
      lastDate
   end
   
   methods
      
      function irs = clsIRS(ric, irsID)
        [tradeDate, settlementDate, maturityDate, floatFixingTimes, floatStartDates, floatQuasiDates, floatPayDates, floatTimeFrac, floatIborEndDates, floatIborTimeFrac, fixedQuasiDates, fixedDates, fixedTimeFrac] = mexPortfolio('cashFlowsIRS', irsID);
        irs.tradeDate = tradeDate;
        irs.maturityDate = maturityDate;
        irs.settlementDate = settlementDate;
        irs.cfDatesFix = fixedDates;
        irs.cfDatesFlt = floatPayDates;
        irs.fixingDatesFlt = floatFixingTimes;
        irs.startDatesFlt = floatStartDates;
        irs.dtFix = fixedTimeFrac;
        irs.dtFlt = floatTimeFrac;

        irs.cfDatesIborFlt = floatIborEndDates;
        irs.dtIborFlt = floatIborTimeFrac;
        irs.assetRIC = ric;

        irs.nFix = length(irs.cfDatesFix);
        irs.nFlt = length(irs.cfDatesFlt);
        
        if (irs.cfDatesFix(end) ~= irs.maturityDate)
          error('This is assumed to determine the last cash flow date');
        end
      end % clsInstruments constructor
              
      function lastDate = get.lastDate(irs)
        lastDate = irs.maturityDate;
      end % Function is called when value is needed
      
      function set.lastDate(irs,~)
         error('You cannot set lastDate explicitly');
      end
      
      function cfDates = getCashFlowDates(irs)
        cfDates = sort([irs.cfDatesFix; irs.cfDatesFlt]);
      end
      
   end
   
   methods (Access = 'private') % Access by class members only
   end
end % classdef

