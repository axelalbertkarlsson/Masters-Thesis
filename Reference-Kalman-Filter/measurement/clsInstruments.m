classdef clsInstruments < handle
   properties
      data = {};
      assetType = {};
      assetRIC = {};
      settlementDate = [];
      maturityDate = [];
      firstDateManuallySet = -1;
   end
   properties (Dependent) % Values that are calculated
      firstDate
      lastDate
      indBill
      indBond
      indCCS
      indCD
      indFRA
      indIBOR
      indIRS
      indOIS
      indTS
   end
   
   methods
      function obj = clsInstruments()
      end % clsInstruments constructor
   end
   
%    methods
%       function td = TensileData(material,samplenum,stress,strain)
%          if nargin > 0
%             td.Material = material;
%             td.SampleNumber = samplenum;
%             td.Stress = stress;
%             td.Strain = strain;
%          end
%       end % TensileData
%    end
   
   methods
%       function obj = set.Material(obj,material)
%       end % Material set function
      
      function l = length(obj)
          l = length(obj.data);
      end

      function firstDate = get.firstDate(obj)
        if (obj.firstDateManuallySet < 0)
          firstDate = min(obj.settlementDate);
        else
          firstDate = obj.firstDateManuallySet;
        end
      end % Function is called when value is needed
      
      function lastDate = get.lastDate(obj)
        lastDate = max(obj.maturityDate);
      end % Function is called when value is needed
      
      function set.firstDate(obj,fDate)
         obj.firstDateManuallySet = fDate;
      end
      
      function set.lastDate(obj,~)
         error('You cannot set lastDate explicitly');
      end

      function indBill = get.indBill(obj)
          indBill = obj.createInd('Bill');
      end % Function is called when value is needed
      
      function indBond = get.indBond(obj)
          indBond = obj.createInd('Bond');
      end % Function is called when value is needed

      function indCCS = get.indCCS(obj)
          indCCS = obj.createInd('CCS');
      end % Function is called when value is needed

      function indCD = get.indCD(obj)
          indCD = obj.createInd('CD');
      end % Function is called when value is needed

      function indFRA = get.indFRA(obj)
          indFRA = obj.createInd('FRA');
      end % Function is called when value is needed

      function indIBOR = get.indIBOR(obj)
          indIBOR = obj.createInd('IBOR');
      end % Function is called when value is needed

      function indIRS = get.indIRS(obj)
          indIRS = obj.createInd('IRS');
      end % Function is called when value is needed

      function indOIS = get.indOIS(obj)
          indOIS = obj.createInd('OIS');
      end % Function is called when value is needed

      function indTS = get.indTS(obj)
          indTS = obj.createInd('TS');
      end % Function is called when value is needed

      function cfDates = getAllCashFlowDates(obj)
          cfDates = [];
          for i = 1:obj.length()
            cfDates = [cfDates; obj.data{i}.getCashFlowDates()];
          end
          cfDates = unique(cfDates);
      end
      
      function add(obj, instr, type, ric, settlement, maturity)
         obj.data{length(obj.data)+1} = instr;
         obj.assetType{length(obj.assetType)+1} = type;
         obj.assetRIC{length(obj.assetRIC)+1} = ric;
         obj.settlementDate = [obj.settlementDate settlement];
         obj.maturityDate = [obj.maturityDate maturity];
      end

      function delete(obj, ind)
          obj.data(ind) = [];
          obj.assetType(ind) = [];
          obj.assetRIC(ind) = [];
          obj.settlementDate(ind) = [];
          obj.maturityDate(ind) = [];
      end
      
%      function disp(obj)
%         fprintf(1,'Instrument class\n');
%      end % This is what is shown when item is displayed in command line
      
%       function plot(td,varargin)
%          plot(td.Strain,td.Stress,varargin{:})
%          title(['Stress/Strain plot for Sample ',num2str(td.SampleNumber)])
%          xlabel('Strain %')
%          ylabel('Stress (psi)')
%       end % plot
   end
   
   % Returns indices to all instruments of the type 'type'
   methods (Access = 'private') % Access by class members only
       function ind = createInd(obj, type)
        ind = zeros(length(obj.assetType),1);
        n = 0;
        for i = 1:length(obj.assetType)
          if (strcmp(obj.assetType{i},type))
            n = n + 1;
            ind(n) = i;
          end
        end
        ind = ind(1:n);        
       end % createInd
   end
end % classdef

