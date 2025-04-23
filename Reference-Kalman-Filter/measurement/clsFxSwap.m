% Class for Interest Rate Swap (IRS)

classdef clsFxSwap < handle
   properties
      assetType = 'fxSwap'
      assetRIC = ''
      
      settlementDate = 0 
      maturityDate = 0
      tradeDate = 0         % The current date for which porperties are set
      price = 0             % Price noted on the trade day
      
      currencyBase = 0
      currencyTerm = 0
      
      settlementBase = 0    % Cash flow at settlement date in base currency
      settlementTerm = 0    % Cash flow at settlement date in term currency
      
      maturityBase = 0      % Cash flow at maturity date in base currency
      maturityTerm = 0      % Cash flow at maturity date in term currency
      
      active = true         % Use this contract for building the curve
      
   end
   
   properties (Dependent) % Values that are calculated
      lastDate
   end
   
   methods
      
      function fxSwap = clsFxSwap(ric, fxSwapID)
        [tradeDate, currencyBase, currencyTerm, settlementDate, settlementBase, settlementTerm, maturityDate, maturityBase, maturityTerm] = mexPortfolio('cashFlowsFxSwap', fxSwapID);
        fxSwap.tradeDate = tradeDate;
        fxSwap.maturityDate = maturityDate;
        fxSwap.settlementDate = settlementDate;
        fxSwap.currencyBase = currencyBase;
        fxSwap.currencyTerm = currencyTerm;
        fxSwap.settlementBase = settlementBase;
        fxSwap.settlementTerm = settlementTerm;
        fxSwap.maturityBase = maturityBase;
        fxSwap.maturityTerm = maturityTerm;
        fxSwap.assetRIC = ric;
       
      end % clsInstruments constructor
              
      function lastDate = get.lastDate(irs)
        lastDate = irs.maturityDate;
      end % Function is called when value is needed
      
      function set.lastDate(fxSwap,~)
         error('You cannot set lastDate explicitly');
      end
      
      function cfDates = getCashFlowDates(fxSwap)
        cfDates = sort([fxSwap.settlemntDate ; fxSwap.maturityDate]);
      end
      
   end
   
   methods (Access = 'private') % Access by class members only
   end
end % classdef

