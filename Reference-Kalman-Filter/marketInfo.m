function [mi] = marketInfo(currency)

% Market conventions for a specific currency
% currency: Currency
% cal: Calendar (defines holidays)
% currencyTimeZone: Time zone for currency (for conversion to UTC time)
% IBOR 
% iborName: RIC for the IBOR used in the most liquid IRS
% iborCal: Calendar (defines holidays) for the IBOR 
% iborCalFixing: Calendar (defines holidays) for the IBOR 
% iborTimeZone: Time zone where IBOR is fixed (for conversion to UTC time)
% iborSettlementLag: Business days between fixing and when interest rate period start for IBOR
% iborBDC: Business Day Convention for IBOR
% iborDCC: Day Counting Convention for IBOR
% irsStructure: Structure for IRS
% ON
% onName: RIC for the IBOR used in the OIS
% onCal: Calendar (defines holidays) for the ON
% onCalFixing: Calendar (defines holidays) for the ON 
% onTenor: Tenor of the over night rate (ON or TN)
% onTimeZone: Time zone where ON is fixed (for conversion to UTC time)
% iborSettlementLag: Business days between fixing and when interest rate period start for ON
% iborBDC: Business Day Convention for ON
% iborDCC: Day Counting Convention for ON
% irsStructure: Structure for ON


mi.currency = currency;

if (currency == 'AUD') % AUD
  mi.cal = 'AUS'; 
  mi.currencyTimeZone = 'Australia/Sydney';

  mi.iborName = 'AU3MEBBR='; 
  mi.iborCal = 'AUD'; 
  mi.iborCalFixing = 'AUS'; 
  mi.iborTenor = '3M'; 
  mi.iborTimeZone = 'Australia/Sydney'; 
  mi.iborSettlementLag = 2; 
  mi.iborBDC = 'M'; 
  mi.iborDCC = 'MMA5';
  mi.irsStructure = 'PAID:FIXED LBOTH SETTLE:1WD CCM:MMA5 DMC:CMF15 EMC:S CFADJ:YES REFDATE:MATURITY CLDR:AUS PDELAY:0 FRQ:4'; % AUD_QM3BB

  mi.onName = ''; 
  mi.onCal = 'AUS'; 
  mi.onCalFixing = 'AUS'; 
  mi.onTenor = 'ON'; 
  mi.onTimeZone = 'Australia/Sydney'; 
  mi.onSettlementLag = 0; 
  mi.onBDC = 'M'; 
  mi.onDCC = 'MMA5';
  mi.oisStructure = 'PAID:FIXED LBOTH SETTLE:0WD FRQ:1 CCM:MMA5 DMC:FOL EMC:SAME CFADJ:NO REFDATE:MATURITY CLDR:AUS PDELAY:1 LFLOAT IDX:OAUDRBA'; % OIS_AUDRBA

  mi.timeToMaturityFRA = '3X6'; 
  mi.fraDCC = 'MMA5';
elseif (currency == 'CAD') % CAD
  mi.cal = 'CAN'; 
  mi.currencyTimeZone = 'America/New_York';
% mi.curencyTimeZone = 'Europe/Paris';  % Bugg gör att tidszonerna inte fungerar korrekt

  mi.iborName = 'CA3MBAFIX='; 
  mi.iborCal = 'CAN'; 
  mi.iborCalFixing = 'CAN'; 
  mi.iborTenor = '3M'; 
  mi.iborTimeZone = 'America/New_York'; 
  mi.iborSettlementLag = 2; 
  mi.iborBDC = 'M'; 
  mi.iborDCC = 'MMA5';
  mi.irsStructure = 'PAID:FIXED LBOTH SETTLE:0WD DMC:M EMC:S CFADJ:YES REFDATE:MATURITY CLDR:CAN PDELAY:0 LFIXED CCM:BBA5 FRQ:2 LFLOAT CCM:MMA5 FRQ:2 RESETFRQ:4'; % CAD_SB3BA
  
  mi.onName = 'CORRA='; 
  mi.onCal = 'CAN'; 
  mi.onCalFixing = 'CAN'; 
  mi.onTenor = 'ON'; 
  mi.onTimeZone = 'America/New_York'; 
  mi.onSettlementLag = 0; 
  mi.onBDC = 'M'; 
  mi.onDCC = 'MMA5';
  mi.oisStructure = 'PAID:FIXED LBOTH SETTLE:1WD FRQ:1 CCM:MMA5 DMC:FOL EMC:SAME CFADJ:NO REFDATE:MATURITY CLDR:CAN PDELAY:1 LFLOAT IDX:OCADCORRA'; % OIS_CADCORRA
  
  mi.timeToMaturityFRA = '3X6'; 
  mi.fraDCC = 'MMA0';
elseif (currency == 'CHF') % CHF
  mi.cal = 'SWI'; 
  mi.currencyTimeZone = 'Europe/Paris';

  mi.iborName = 'LIBORCHF3M'; 
  mi.iborCal = 'SWI,UKG'; 
  mi.iborCalFixing = 'UKG'; 
  mi.iborTenor = '6M'; 
  mi.iborTimeZone = 'Europe/London'; 
  mi.iborSettlementLag = 2; 
  mi.iborBDC = 'M'; 
  mi.iborDCC = 'MMA0';
  mi.irsStructure = 'PAID:FIXED LBOTH SETTLE:2WD DMC:M EMC:S CFADJ:YES REFDATE:MATURITY CLDR:SWI,UKG PDELAY:0 LFIXED FRQ:1 CCM:BB00 LFLOAT FRQ:2 CCM:MMA0'; % CHF_AB6L

  mi.onName = 'SARON.S'; 
  mi.onCal = 'SWI'; 
  mi.onCalFixing = 'SWI'; 
  mi.onTenor = 'ON'; 
  mi.onTimeZone = 'Europe/Paris'; 
  mi.onSettlementLag = 0; 
  mi.onBDC = 'M'; 
  mi.onDCC = 'MMA0';
  mi.oisStructure = 'PAID:FIXED LBOTH SETTLE:2WD FRQ:1 CCM:MMA0 DMC:FOL EMC:SAME CFADJ:NO REFDATE:MATURITY CLDR:SWI PDELAY:0 LFLOAT IDX:OCHFTOIS'; % OIS_CHFTOIS

  mi.timeToMaturityFRA = '6X12'; 
  mi.fraDCC = 'MMA0';
elseif (currency == 'EUR') % EUR
  mi.cal = 'EMU'; 
  mi.currencyTimeZone = 'Europe/Paris';

  mi.iborName = 'EURIBOR6M'; 
  mi.iborCal = 'EMU'; 
  mi.iborCalFixing = 'EUR'; 
  mi.iborTenor = '6M'; 
  mi.iborTimeZone = 'Europe/Paris'; 
  mi.iborSettlementLag = 2; 
  mi.iborBDC = 'M'; 
  mi.iborDCC = 'MMA0';
  mi.irsStructure = 'PAID:FIXED LBOTH SETTLE:2WD DMC:M EMC:S CFADJ:YES REFDATE:MATURITY CLDR:EMU PDELAY:0 LFIXED FRQ:1 CCM:BB00 LFLOAT FRQ:2 CCM:MMA0'; % EUR_AB6E
  
  mi.onName = 'EONIA='; 
  mi.onCal = 'EMU'; 
  mi.onCalFixing = 'EMU'; 
  mi.onTenor = 'ON'; 
  mi.onTimeZone = 'Europe/Paris'; 
  mi.onSettlementLag = 0; 
  mi.onBDC = 'M'; 
  mi.onDCC = 'MMA0';
  mi.oisStructure = 'PAID:FIXED LBOTH SETTLE:0WD FRQ:1 CCM:MMA0 DMC:FOL EMC:SAME CFADJ:NO REFDATE:MATURITY CLDR:EMU PDELAY:1 LFLOAT IDX:OEONIA'; % OIS_EONIA

  mi.timeToMaturityFRA = '6X12'; 
  mi.fraDCC = 'MMA0';
elseif (currency == 'GBP') % GBP, Check yearly time frac
  mi.cal = 'UKG'; 
  mi.currencyTimeZone = 'Europe/London';

  mi.iborName = 'LIBORGBP6M'; 
  mi.iborCal = 'UKG'; 
  mi.iborCalFixing = 'UKG'; 
  mi.iborTenor = '6M'; 
  mi.iborTimeZone = 'Europe/London'; 
  mi.iborSettlementLag = 0; 
  mi.iborBDC = 'M'; 
  mi.iborDCC = 'MMA5';
  mi.irsStructure = 'PAID:FIXED LBOTH SETTLE:0WD FRQ:2 DMC:M EMC:S CFADJ:YES REFDATE:MATURITY CLDR:UKG PDELAY:0 LFIXED CCM:BBA5 LFLOAT CCM:MMA5'; % GBP_SB6L (fix)
  
  mi.onName = 'SONIAOSR='; 
  mi.onCal = 'UKG'; 
  mi.onCalFixing = 'UKG'; 
  mi.onTenor = 'ON'; 
  mi.onTimeZone = 'Europe/London'; 
  mi.onSettlementLag = 0; 
  mi.onBDC = 'M'; 
  mi.onDCC = 'MMA5';
  mi.oisStructure = 'PAID:FIXED LBOTH SETTLE:0WD FRQ:1 CCM:MMA5 DMC:FOL EMC:SAME CFADJ:NO REFDATE:MATURITY CLDR:EMU PDELAY:0 LFLOAT IDX:OSONIA'; % OIS_SONIA

  mi.timeToMaturityFRA = '6X12'; 
  mi.fraDCC = 'MMA5';
elseif (currency == 'JPY') % JPY
  mi.cal = 'JAP'; 
  mi.currencyTimeZone = 'Asia/Tokyo';
  
  mi.iborName = 'LIBORJPY6M'; 
  mi.iborCal = 'JAP,UKG'; 
  mi.iborCalFixing = 'UKG'; 
  mi.iborTenor = '6M'; 
  mi.iborTimeZone = 'Europe/London'; 
  mi.iborSettlementLag = 2; 
  mi.iborBDC = 'M'; 
  mi.iborDCC = 'MMA0';
  mi.irsStructure = 'PAID:FIXED LBOTH SETTLE:2WD FRQ:2 DMC:M EMC:S CFADJ:YES REFDATE:MATURITY CLDR:JAP,UKG PDELAY:0 LFIXED CCM:BBA5 LFLOAT CCM:MMA5'; % JPY_SB6L (fix)
  
  mi.onName = 'JPONMU=RR'; 
  mi.onCal = 'JAP'; 
  mi.onCalFixing = 'JAP'; 
  mi.onTenor = 'ON'; 
  mi.onTimeZone = 'Asia/Tokyo'; 
  mi.onSettlementLag = 0; 
  mi.onBDC = 'M'; 
  mi.onDCC = 'MMA5';
  mi.oisStructure = 'PAID:FIXED LBOTH SETTLE:2WD FRQ:1 CCM:MMA5 DMC:FOL EMC:SAME CFADJ:NO REFDATE:MATURITY CLDR:JAP PDELAY:2 LFLOAT IDX:OJPYONMU'; % OIS_JPYONMU

  mi.timeToMaturityFRA = '6X12'; 
  mi.fraDCC = 'MMA0';
elseif (currency == 'KRW') % KRW
  mi.cal = 'KOR'; 
  mi.currencyTimeZone = 'Asia/Seoul';
  
  mi.iborName = 'KRWCD3M'; 
  mi.iborCal = 'KOR'; 
  mi.iborCalFixing = 'KOR'; 
  mi.iborTenor = '3M'; 
  mi.iborTimeZone = 'Asia/Seoul'; 
  mi.iborSettlementLag = 2; 
  mi.iborBDC = 'M'; 
  mi.iborDCC = 'MMA5';
  mi.irsStructure = 'PAID:FIXED LBOTH SETTLE:1WD CCM:MMA5 DMC:MOD EMC:S CFADJ:YES REFDATE:MATURITY PDELAY:0 CLDR:KOR LFIXED FRQ:4 LFLOAT FRQ:4'; % KRW_QMCD
  
%   clear onName; % Do hot have OIS
%   clear oisStructure; % Do hot have OIS
  
  mi.timeToMaturityFRA = '3X6'; 
  mi.fraDCC = 'MMA5';

elseif (currency == 'NOK') % SEK
  mi.cal = 'NOR'; 
  mi.currencyTimeZone = 'Europe/Oslo';
  
  mi.iborName = 'OINOK6MD='; 
  mi.iborCal = 'NOR'; 
  mi.iborCalFixing = 'NOR'; 
  mi.iborTenor = '6M'; 
  mi.iborTimeZone = 'Europe/Oslo'; 
  mi.iborSettlementLag = 2; 
  mi.iborBDC = 'M'; 
  mi.iborDCC = 'MMA0';
  mi.irsStructure = 'PAID:FIXED LBOTH SETTLE:2WD DMC:M EMC:S CFADJ:YES REFDATE:MATURITY CLDR:NOR PDELAY:0 LFIXED FRQ:1 CCM:BB00 LFLOAT FRQ:2 CCM:MMA0'; % NOK_AB6O
  
  mi.onName = 'NOWA='; 
  mi.onCal = 'NOR'; 
  mi.onCalFixing = 'NOR'; 
  mi.onTenor = 'ON'; 
  mi.onTimeZone = 'Europe/Oslo'; 
  mi.onSettlementLag = 0; 
  mi.onBDC = 'M'; 
  mi.onDCC = 'MMA0';
  mi.oisStructure = 'PAID:FIXED LBOTH SETTLE:2WD FRQ:1 CCM:MMA5 DMC:M EMC:SAME CFADJ:NO REFDATE:MATURITY CLDR:NOR PDELAY:1 LFLOAT IDX:ONOWA'; % 

  mi.timeToMaturityFRA = '3F1'; 
  mi.fraDCC = 'MMA0';

elseif (currency == 'SEK') % SEK
  mi.cal = 'SWE'; 
  mi.currencyTimeZone = 'Europe/Stockholm';
  
  mi.iborName = 'STIBOR3M'; 
  mi.iborCal = 'SWE'; 
  mi.iborCalFixing = 'SWE'; 
  mi.iborTenor = '3M'; 
  mi.iborTimeZone = 'Europe/Stockholm'; 
  mi.iborSettlementLag = 2; 
  mi.iborBDC = 'M'; 
  mi.iborDCC = 'MMA0';
  mi.irsStructure = 'PAID:FIXED LBOTH SETTLE:2WD DMC:M EMC:S CFADJ:YES REFDATE:MATURITY CLDR:SWE PDELAY:0 LFIXED FRQ:1 CCM:BB00 LFLOAT FRQ:4 CCM:MMA0'; % SEK_AB3S
  
  mi.onName = 'STISEKTNDFI='; 
  mi.onCal = 'SWE'; 
  mi.onCalFixing = 'SWE'; 
  mi.onTenor = 'TN'; 
  mi.onTimeZone = 'Europe/Stockholm'; 
  mi.onSettlementLag = 0; 
  mi.onBDC = 'M'; 
  mi.onDCC = 'MMA0';
  mi.oisStructure = 'PAID:FIXED LBOTH SETTLE:2WD FRQ:1 CCM:MMA0 DMC:FOL EMC:SAME CFADJ:NO REFDATE:MATURITY CLDR:SWE PDELAY:0 LFLOAT IDX:OSEKSTI'; % OIS_SEKSTI

  mi.timeToMaturityFRA = '3F1'; 
  mi.fraDCC = 'MMA0';
elseif (currency == 'USD') % USD
  mi.cal = 'USA'; 
  mi.currencyTimeZone = 'America/New_York';
% mi.curencyTimeZone = 'Europe/Paris';  % Bugg gör att tidszonerna inte fungerar korrekt

  mi.iborName = 'LIBORUSD3M'; 
  mi.iborCal = 'USA,UKG'; 
  mi.iborCalFixing = 'UKG'; 
  mi.iborTenor = '3M'; 
  mi.iborTimeZone = 'Europe/London'; 
  mi.iborSettlementLag = 2; 
  mi.iborBDC = 'M'; 
  mi.iborDCC = 'MMA0';
  mi.irsStructure = 'PAID:FIXED LBOTH SETTLE:2WD CCM:MMA0 DMC:M EMC:S CFADJ:YES REFDATE:MATURITY CLDR:USA PDELAY:0 LFIXED FRQ:1 LFLOAT FRQ:4'; % USD_AM3L
  
  mi.onName = 'USONFFE='; 
  mi.onCal = 'USA'; 
  mi.onCalFixing = 'USA'; 
  mi.onTenor = 'ON'; 
  mi.onTimeZone = 'America/New_York'; 
  mi.onSettlementLag = 0; 
  mi.onBDC = 'M'; 
  mi.onDCC = 'MMA0';
  mi.oisStructure = 'PAID:FIXED LBOTH SETTLE:2WD FRQ:1 CCM:MMA0 DMC:FOL EMC:SAME CFADJ:NO REFDATE:MATURITY CLDR:USA PDELAY:2 LFLOAT IDX:OFFER'; % 
  
  mi.timeToMaturityFRA = '3X6'; 
  mi.fraDCC = 'MMA0';
else
  error('Unknown currency');
end
