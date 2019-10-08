
#include <iostream>
#include <iomanip>
//#include <stdlib> //for EXIT_FAILURE
#include <cmath>
// #include "Math/PdfFuncMathCore.h"  //tried to fix "Error: Function gaussian_cdf(significance,1) is not defined in Pvalue_uncapped.C
// #include "Math/ProbFuncMathCore.h" //tried to fix "Error: Function gaussian_cdf(significance,1) is not defined in Pvalue_uncapped.C
#include <fstream>

#include "TH1.h"
#include "TString.h"
#include "TFile.h"
#include "TF1.h"
#include "TLine.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TGraphErrors.h"
#include "TVirtualFitter.h"
#include "TLegend.h"
#include "TMath.h"

//#include "AtlasStyle.C"
#include "get_signalmodel.C"
//#include "signalRate.C" 
//#include "createTable.h"
//#include "Pvalue.C" //chi-squared
#include "Pvalue_uncapped.C"

using std::cout;
using std::endl;

TH1 *h_hist;

//Plot with the expected and observed p0 and significance
TH1F *hp0_expected_asimov;
TH1F *hp0_observed_asimov;
TH1F *hsignificance_expected_asimov;
TH1F *hsignificance_observed_asimov;

TH1D *htoys;


TF1 *Bernstein4(Double_t xmin=100, Double_t xmax=160);

//float number_of_signal_events(TString &histname, double luminosity);
void FixParametersOfSpuriousModel(int number_of_background_parameters, TF1 *model, double binwidth, double alphaCB, double sigmaG, double meanG, double fractionCB, double sigmaCB, double meanCB);
void PrintPerformance(Double_t mass, Double_t &expectedSignal, Double_t &spuriousSignal, Double_t &percentS, Double_t &percentB, Double_t &sigma0, Double_t &significance, Double_t &significanceBias, Double_t &delta, Double_t &sigma0prime, Double_t & significanceCovered, Double_t & chisquared, bool printToScreen, ofstream &file);


int HSG1BackgroundBiasStudy(Double_t minmass=100,Double_t maxmass=160,TString filename="diphox_shape_withGJJJDY_WithEffCor.root",TString histname="Mgg_CP1", TString backgroundfunc = "exp", int number_of_toys=400, Double_t luminosity=4.9){
  
  //gROOT->LoadMacro("createTable.cc");
  //gROOT->LoadMacro("signalRate.C");

  //Options: 
  bool writeResultToFiles = 0;
  bool dologlikelihood = 0;
  bool verbose = 1;
  bool backgroundAsimovInput=1;
  bool turnOffSpurious = 0;
  bool minuit2=0;
  bool minos=0;
  bool improve=0;
  bool showfitbands=1;
  bool rebinInputHisto = 0;

  cout << endl 
       <<"--- Options ---" << endl
       << "verbose:                " << verbose << endl
       << "rebinInputHisto:        " << rebinInputHisto << endl
       << "dologlikelihood:        " << dologlikelihood << endl
       << "backgroundAsimovInput:  " << backgroundAsimovInput << endl
       << "turnOffSpurious:        " << turnOffSpurious << endl
       << "minuit2:                " <<  minuit2 << endl
       << "minos:                  " << minos << endl
       << "improve:                " << improve << endl 
       << "showfitbands:           " << showfitbands << endl
       << "writeResultToFiles:     " << writeResultToFiles << endl
              << endl;

  Int_t failedFitsMC=0;
  Int_t failedFitsAsimov=0;
  Int_t failedFitsToys=0;
  Int_t errorMatrixProblemsMC=0;
  Int_t errorMatrixProblemsAsimov=0;
  Int_t errorMatrixProblemsToys=0;

  //SetAtlasStyle();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //gStyle->SetTitleOffset(0.7,"Y");
 
  //Only affects the second pad, no idea why.. (But the first histogram luckily already has Helvetica, 0.05 it seems)
  Int_t font=42; // Helvetica
  Double_t tsize=0.05;
  gStyle->SetTextFont(font);
  gStyle->SetTextSize(tsize);
 
  gStyle->SetLabelFont(font,"x");
  gStyle->SetTitleFont(font,"x");
  gStyle->SetLabelFont(font,"y");
  gStyle->SetTitleFont(font,"y");
  
  gStyle->SetLabelSize(tsize,"x");
  gStyle->SetTitleSize(tsize,"x");
  gStyle->SetLabelSize(tsize,"y");
  gStyle->SetTitleSize(tsize,"y");
  
  //gROOT->ForceStyle(); //messes up the marker size
  


  cout << endl;
  cout << "Your choice of background models: " << endl;
  cout << "---------------------------------------------" << endl;
  cout << "\"exp\"           - exponential" << endl;
  cout << "\"expw\"          - exponential window cut" << endl;
  cout << "\"doubleexp\"     - double exponential" << endl;
  cout << "\"expturnon\"     - exponential*turnon" << endl;
  cout << "\"expturnonexp\"  - exponential*turnon + exponential" << endl;
  cout << "\"epoly2\"        - exponential(polynomial2)" << endl;
  cout << "\"epoly2w\"       - exponential(polynomial2) window cut" << endl;
  cout << "\"epoly3\"        - exponential(polynomial3)" << endl;
  cout << "\"epoly3w\"       - exponential(polynomial3) window cut" << endl;
  cout << "\"epoly4\"        - exponential(polynomial4)" << endl;
  cout << "\"epoly5\"        - exponential(polynomial5)" << endl;
  cout << "\"eturnonl\"      - exponential*turnon + linear" << endl;
  cout << "\"bern3\"         - 3rd order Bernstein" << endl;
  cout << "\"bern3w\"         - rd order Bernstein window cut" << endl;
  cout << "\"bern4\"         - 4th order Bernstein" << endl;
  cout << "\"bern4w\"        - 4th order Bernstein window cut" << endl;
  cout << "\"bern5\"         - 5th order Bernstein" << endl;
  cout << "\"bern5w\"        - 5th order Bernstein window cut" << endl;
  cout << "\"bern6\"         - 6th order Bernstein" << endl;
  cout << "\"bern6w\"        - 6th order Bernstein window cut" << endl;
  cout << "\"bern7\"         - 7th order Bernstein" << endl;
  cout << "\"bern7w\"        - 7th order Bernstein window cut" << endl;
  cout << "---------------------------------------------" << endl << endl;



  //TFile *tf=new TFile(filename,"r");
  TFile *tf = TFile::Open(filename);
  if(!tf) {cout << "Could not read file " << filename << "!"; return(0);} // exit(EXIT_FAILURE);}
  tf->cd();
  tf->ReadAll();

  h_hist = (TH1 *)tf->Get(histname)->Clone();
  h_hist->Scale(luminosity/4.9); // Scale to luminosity
  if(h_hist->IsZombie() || h_hist==NULL){
    cout << "Histogram " << histname << " not found!" << endl; return(0);} // exit(EXIT_FAILURE); }
  
  float binwidth = h_hist->GetBinWidth(1);
  Int_t nbins=h_hist->GetXaxis()->GetNbins();
  Double_t xmin = h_hist->GetXaxis()->GetXmin();
  Double_t xmax = h_hist->GetXaxis()->GetXmax();
  cout << nbins << " HIST! " << xmin << " " << xmax << endl;



  //Find the correct category in question:
  int    category;
  if(histname.Length()>22) //hist_InvariantMass_1GeV_100_160_standardPtcuts (hist_InvariantMass_CP1 has length 22)
    category = 0; 
  else if(histname.Contains("10"))
    category = 10;
  else if (histname.Contains("weighted")) // for this, a weighted inclusive signal model is built in get_signalmodel, and in signalRate, the inclusive is used
    category = 11; 
  else if(histname.Contains("0"))
    category = 0;
  else if(histname.Contains("1"))
    category = 1;
  else if(histname.Contains("2"))
    category = 2;
  else if(histname.Contains("3"))
    category = 3;
  else if(histname.Contains("4"))
    category = 4;
  else if(histname.Contains("5"))
    category = 5;
  else if(histname.Contains("6"))
    category = 6;
  else if(histname.Contains("7"))
    category = 7;
  else if(histname.Contains("8"))
    category = 8;
  else if(histname.Contains("9"))
    category = 9;




  //Rebinning input histogram
  double desiredGeVperBin;
  if(rebinInputHisto) {
    if(category==0)
      desiredGeVperBin = 2;
    if(category==1)
      desiredGeVperBin = 1;  
    if(category==2)
      desiredGeVperBin = 10;  
    if(category==3)
      desiredGeVperBin = 0.5;  
    if(category==4)
      desiredGeVperBin = 4;      
    if(category==5)
      desiredGeVperBin = 2;      
    if(category==6)
      desiredGeVperBin = 10;  
    if(category==7)
      desiredGeVperBin = 0.5;  
    if(category==8)
      desiredGeVperBin = 2;  
    if(category==9)
      desiredGeVperBin = 0.5;  
    int rebinfactor_originhisto = (int)(desiredGeVperBin/((xmax-xmin)/nbins));
    h_hist->Rebin(rebinfactor_originhisto);
    nbins=h_hist->GetXaxis()->GetNbins(); //new nbins
    binwidth = h_hist->GetBinWidth(1); //new binwidth
  }

  int numbins100160 = (160-100)/binwidth; //used for expected RMS of pulls distribution
  //cout << "numbins100160 = " << numbins100160 << endl;


  // Blind the 2011 "possible signal" window from 120 to 130
  if(backgroundfunc.Contains("w")) {
    for(Int_t ibin=1;ibin<=nbins;ibin++) {
      if(h_hist->GetBinCenter(ibin) >=120 && h_hist->GetBinCenter(ibin)<=130) {
  	h_hist->SetBinContent(ibin,0);
  	h_hist->SetBinError(ibin,0);
	cout << "Zero bin " << ibin << endl;
      }
    }
  }
 





  //To feed into plotSign.C to get the significance plot as made by Diego and Georgios
  TString significanceRootfileName = "BackgroundFitAndData_CP";
  significanceRootfileName += category;
  significanceRootfileName += backgroundfunc;
  significanceRootfileName += ".root";
  TFile *significanceRootfile = new TFile(significanceRootfileName,"RECREATE");
  significanceRootfile->cd();

  h_hist->Write("data");

  //Histogram with the expectation from the background-only fit to the sample (for significanceRootfile)
  // will be filled with the background fit to the input histogram
  TH1* hbackgroundExpectation = (TH1 *)h_hist->Clone("hbackgroundExpectation");
  hbackgroundExpectation->Reset();


  //Expected signal rate
  TH1D *hexpectedSignal = new TH1D("hexpectedSignal","Signal rate vs mH",nbins,xmin,xmax);
  //signalRate(category,hexpectedSignal,luminosity);


  //Globals: 
  hp0_expected_asimov = new TH1F("hp0_expected_asimov","Expected p0 from Asimov vs invariant mass",nbins, xmin, xmax);
  hp0_observed_asimov = new TH1F("hp0_observed_asimov","Observed p0 from Asimov vs invariant mass",nbins, xmin, xmax);
  hsignificance_expected_asimov = new TH1F("hsignificance_expected_asimov","Expected significance from Asimov vs invariant mass",nbins, xmin, xmax);
  hsignificance_observed_asimov = new TH1F("hsignificance_observed_asimov","Observed significance from Asimov vs invariant mass",nbins, xmin, xmax);


  //For convergence of tricky fits, need to increase the number of times Minuit is called
  int MaxNumberOfCallsToMinuit = 5000; //default is 5000
  TVirtualFitter::SetMaxIterations( MaxNumberOfCallsToMinuit );
  if(minuit2) TVirtualFitter::SetDefaultFitter("Minuit2"); 
  //TVirtualFitter::SetPrecision(2E-5); // 2E-5 minimizes crappy Asimov errors for CP1 epoly2
  //TVirtualFitter::SetPrecision(2E-12);


  // "Smart pointer" to fit results
  TFitResultPtr fitresult,backgroundfitresult;
  Int_t fitstatus=0;


  //-------- Define all the background functions, choose which one to use (TF1 *backgroundfunction) underneath ----------


  //Exponential function
  // is of the form: "binwidth*[0]*[1]/(exp(-[1]*minmass)-exp(-[1]*maxmass))*exp(-[1]*x)"  

  // With this form, we can interpret [0] as the number of events (and [1] as the slope)
  // -- we have to make up for that exp(-beta*x) is not a unit pdf, by compensating with a number, lambda:
  // 1 = integral_minmass^maxmass {lambda*exp(-beta*x)} dx ==> lambda = beta/(exp(-beta*minmass) - exp(-beta*maxmass))

  //Must build a string, because Root does not understand expressions like "binwidth" in the formula
  TString exponential = "";
  exponential += binwidth;
  exponential += "*[0]*[1]/(exp(-[1]*";
  exponential += minmass;
  exponential += ")-exp(-[1]*";
  exponential += maxmass;
  exponential += "))*exp(-[1]*x)";
  if(backgroundfunc == "expw") exponential = "(x<120)*" + exponential + " + " + exponential + "*(x>130)";
  
  TF1 *expDist=new TF1("expDist",exponential,minmass,maxmass);
  //float betaguess = 0.025;
  float betaguess = 0.00275; //testing different value for 2011 CP10, which is almost flat (number gotten from Jana)
  //float betaguess =  1/h_hist->GetMean(); //slope is 1/mean for an exponential distribution from -infty to +infty
  expDist->SetParameter(0,h_hist->GetSumOfWeights()); 
  expDist->SetParameter(1,betaguess); 



  //Double exponential
  // "doubleExpDist","binwidth*[0]*[1]/(exp(-[1]*minmass)-exp(-[1]*maxmass))*exp(-[1]*x) + binwidth*[2]*[3]/(exp(-[3]*minmass)-exp(-[3]*maxmass))*exp(-[3]*x)"
  TString doubleExponential = exponential;
  doubleExponential += "+";
  doubleExponential += binwidth;
  doubleExponential += "*[2]*[3]/(exp(-[3]*";
  doubleExponential += minmass;
  doubleExponential += ")-exp(-[3]*";
  doubleExponential += maxmass;
  doubleExponential += "))*exp(-[3]*x)";
  //cout << "doubleeExponential = " << doubleExponential << endl;
  
  TF1 *doubleExpDist=new TF1("doubleExpDist",doubleExponential,minmass,maxmass);
  doubleExpDist->SetParameter(0,h_hist->GetSumOfWeights());
  doubleExpDist->SetParameter(1,betaguess);
  doubleExpDist->SetParameter(2,h_hist->GetSumOfWeights()/10);
  // float doublebetaguess = 0.3;
  // doubleExpDist->SetParameter(3,doublebetaguess);
  doubleExpDist->SetParameter(3,betaguess*3);

  //Original setup:
  // TF1 *doubleExpDist=new TF1("doubleExpDist","[0]*exp(-[1]*x)+[2]*exp(-[3]*x)",minmass,maxmass);
  // expDist->SetParameter(0,300);
  // expDist->SetParameter(1,0.025);
  // expDist->SetParameter(2,0.2);
  // expDist->SetParameter(3,0.3);

  //Exponential times turn-on
  TF1 *expTurnOnDist=new TF1("expTurnOnDist","[0]*exp(-[1]*x)/(1+[2]*exp(-[3]*(x-100)))",minmass,maxmass);
  expTurnOnDist->SetParameter(0,1100);
  expTurnOnDist->SetParameter(1,0.025);
  expTurnOnDist->SetParameter(2,0.05);
  expTurnOnDist->SetParameter(3,0.15);
  expTurnOnDist->SetParLimits(2,0,0.2); // Limits so that turn-on form is kept
  expTurnOnDist->SetParLimits(3,0.05,1);

  //Exponential times turn-on plus exponential
  TF1 *expTurnOnExpDist=new TF1("expTurnOnExpDist","[0]*exp(-[1]*x)/(1+[2]*exp(-[3]*(x-100)))+[4]*exp(-[5]*x)",minmass,maxmass);

  // Exponential times turn-on plus linear
  TF1 *eturnonl = new TF1("eturnonl","[0]*exp(-[1]*x)/(1+[2]*exp(-[3]*(x-100)))+[4]+[5]*(x-100)",xmin,xmax);
  Double_t eturnonlStartParameters[6];
  eturnonl->SetParameter(0,1100);
  eturnonl->SetParameter(1,0.025);
  eturnonl->SetParameter(2,0.05);
  eturnonl->SetParameter(3,0.15);
  eturnonl->SetParLimits(2,0,0.2); // Limits so that turn-on form is kept
  eturnonl->SetParLimits(3,0.05,1);
  eturnonl->SetParameter(4,0);
  eturnonl->SetParameter(5,0);
  eturnonl->GetParameters(eturnonlStartParameters);

  TF1 *epoly2=new TF1("epoly2","exp([0]+[1]*(x-100)+[2]*pow(x-100,2))",minmass,maxmass);
  TF1 *epoly2w=new TF1("epoly2w","(x<120)*exp([0]+[1]*(x-100)+[2]*pow(x-100,2))+(x>130)*exp([0]+[1]*(x-100)+[2]*pow(x-100,2))",minmass,maxmass);
  TF1 *epoly3=new TF1("epoly3","exp([0]+[1]*(x-100)+[2]*pow(x-100,2)+[3]*pow(x-100,3))",minmass,maxmass);
  TF1 *epoly3w=new TF1("epoly3w","(x<120)*exp([0]+[1]*(x-100)+[2]*pow(x-100,2)+[3]*pow(x-100,3))+(x>130)*exp([0]+[1]*(x-100)+[2]*pow(x-100,2)+[3]*pow(x-100,3))",minmass,maxmass);
  TF1 *epoly4=new TF1("epoly4","exp([0]+[1]*(x-100)+[2]*pow(x-100,2)+[3]*pow(x-100,3)+[4]*pow(x-100,4))",minmass,maxmass);
  TF1 *epoly5=new TF1("epoly5","exp([0]+[1]*(x-100)+[2]*pow(x-100,2)+[3]*pow(x-100,3)+[4]*pow(x-100,4)+[5]*pow(x-100,5))",minmass,maxmass);

  //Make pointer to the preferred background model
  TF1 *backgroundfunction;
  if(backgroundfunc=="exp" || backgroundfunc=="expw" ){
    backgroundfunction = expDist;
  }  
  else if(backgroundfunc=="doubleexp"){
    backgroundfunction = doubleExpDist;
  }  
  else if(backgroundfunc=="expturnon"){
    backgroundfunction = expTurnOnDist;
  }  
  else if(backgroundfunc=="expturnonexp"){
    backgroundfunction = expTurnOnExpDist;
  }
  else if(backgroundfunc=="epoly2"){
    backgroundfunction = epoly2;
    backgroundfunction->SetParameter(0,2);
    backgroundfunction->SetParameter(1,-0.03);
    backgroundfunction->SetParameter(2,0);
  }
  else if(backgroundfunc=="epoly2w"){
    backgroundfunction = epoly2w;
    backgroundfunction->SetParameter(0,2);
    backgroundfunction->SetParameter(1,-0.03);
    backgroundfunction->SetParameter(2,0);
  }
  else if(backgroundfunc=="epoly3"){
    backgroundfunction = epoly3;
    backgroundfunction->SetParameter(0,2);
    backgroundfunction->SetParameter(1,-0.03);
    backgroundfunction->SetParameter(2,0);
    backgroundfunction->SetParameter(3,0);
  }
  else if(backgroundfunc=="epoly3w"){
    backgroundfunction = epoly3w;
    backgroundfunction->SetParameter(0,2);
    backgroundfunction->SetParameter(1,-0.03);
    backgroundfunction->SetParameter(2,0);
    backgroundfunction->SetParameter(3,0);
  }
  else if(backgroundfunc=="epoly4"){
    backgroundfunction = epoly4;
   // 1  p0           4.05796e+00   4.56923e-04   0.00000e+00  -9.60617e-05
   // 2  p1          -2.78265e-02   1.31752e-04  -0.00000e+00   3.33575e-03
   // 3  p2           1.82736e-04   1.00058e-05   0.00000e+00  -7.24732e-02
   // 4  p3          -3.37125e-06   2.68172e-07  -0.00000e+00  -5.55895e+00
   // 5  p4           2.51295e-08   2.31675e-09   0.00000e+00  -3.47559e+02
    backgroundfunction->SetParameter(0,4.1);
    backgroundfunction->SetParameter(1,-2e-2);
    backgroundfunction->SetParameter(2,2e-4);
    backgroundfunction->SetParameter(3,-3e-6);
    backgroundfunction->SetParameter(4,2e-8);
  }
  else if(backgroundfunc=="epoly5"){
    backgroundfunction = epoly5;
    backgroundfunction->SetParameter(0,2);
    backgroundfunction->SetParameter(1,-0.03);
    backgroundfunction->SetParameter(2,0);
    backgroundfunction->SetParameter(3,0);
    backgroundfunction->SetParameter(4,0);
    backgroundfunction->SetParameter(5,0);
    backgroundfunction->SetParameter(6,0);
  }
  else if(backgroundfunc=="eturnonl"){
    backgroundfunction = eturnonl;
  }
  else if(backgroundfunc=="bern3" || backgroundfunc=="bern3w"){
    TString fstring= "[0]*1*pow((x-[4])/([5]-[4]),0)*pow(([5]-x)/([5]-[4]),3)";
    fstring += "+ [1]*3*pow((x-[4])/([5]-[4]),1)*pow(([5]-x)/([5]-[4]),2)";
    fstring += "+ [2]*3*pow((x-[4])/([5]-[4]),2)*pow(([5]-x)/([5]-[4]),1)";
    fstring += "+ [3]*1*pow((x-[4])/([5]-[4]),3)*pow(([5]-x)/([5]-[4]),0)";
    
    if(backgroundfunc=="bern3w") fstring = "(x<120)*(" + fstring + ") + (x>130)*(" + fstring + ")";
    backgroundfunction = new TF1("Bernstein3",fstring,xmin,xmax);
    backgroundfunction->SetParameters(1,0.1,0.01,0.001,0.0001);
    backgroundfunction->SetParNames("c0","c1","c2","c3","xmin","xmax");
    backgroundfunction->FixParameter(4,minmass);
    backgroundfunction->FixParameter(5,maxmass);
  }
  else if(backgroundfunc=="bern4" || backgroundfunc=="bern4w"){
    TString fstring= "[0]*1*pow((x-[5])/([6]-[5]),0)*pow(([6]-x)/([6]-[5]),4)";
    fstring += "+ [1]*4*pow((x-[5])/([6]-[5]),1)*pow(([6]-x)/([6]-[5]),3)";
    fstring += "+ [2]*6*pow((x-[5])/([6]-[5]),2)*pow(([6]-x)/([6]-[5]),2)";
    fstring += "+ [3]*4*pow((x-[5])/([6]-[5]),3)*pow(([6]-x)/([6]-[5]),1)";
    fstring += "+ [4]*1*pow((x-[5])/([6]-[5]),4)*pow(([6]-x)/([6]-[5]),0)";
    
    if(backgroundfunc=="bern4w") fstring = "(x<120)*(" + fstring + ") + (x>130)*(" + fstring + ")";
    backgroundfunction = new TF1("Bernstein4",fstring,xmin,xmax);
    backgroundfunction->SetParameters(1,0.1,0.01,0.001,0.0001);
    backgroundfunction->SetParNames("c0","c1","c2","c3","c4","xmin","xmax");
    backgroundfunction->FixParameter(5,minmass);
    backgroundfunction->FixParameter(6,maxmass);
  }
  else if(backgroundfunc=="bern5" ||backgroundfunc=="bern5w"){
    TString fstring= "[0]*1*pow((x-[6])/([7]-[6]),0)*pow(([7]-x)/([7]-[6]),5)";
    fstring += "+ [1]*5*pow((x-[6])/([7]-[6]),1)*pow(([7]-x)/([7]-[6]),4)";
    fstring += "+ [2]*10*pow((x-[6])/([7]-[6]),2)*pow(([7]-x)/([7]-[6]),3)";
    fstring += "+ [3]*10*pow((x-[6])/([7]-[6]),3)*pow(([7]-x)/([7]-[6]),2)";
    fstring += "+ [4]*5*pow((x-[6])/([7]-[6]),4)*pow(([7]-x)/([7]-[6]),1)";
    fstring += "+ [5]*1*pow((x-[6])/([7]-[6]),5)*pow(([7]-x)/([7]-[6]),0)";
    
    if(backgroundfunc=="bern5w") fstring = "(x<120)*(" + fstring + ") + (x>130)*(" + fstring + ")";
    backgroundfunction = new TF1("Bernstein5",fstring,xmin,xmax);
    backgroundfunction->SetParameters(1,0.1,0.01,0.001,0.0001,0.00001);
    backgroundfunction->SetParNames("c0","c1","c2","c3","c4","c5","xmin","xmax");
    backgroundfunction->FixParameter(6,minmass);
    backgroundfunction->FixParameter(7,maxmass);
  }
  else if(backgroundfunc=="bern6" || backgroundfunc=="bern6w"){
    TString fstring= "[0]*1*pow((x-[7])/([8]-[7]),0)*pow(([8]-x)/([8]-[7]),6)";
    fstring += "+ [1]*6*pow((x-[7])/([8]-[7]),1)*pow(([8]-x)/([8]-[7]),5)";
    fstring += "+ [2]*15*pow((x-[7])/([8]-[7]),2)*pow(([8]-x)/([8]-[7]),4)";
    fstring += "+ [3]*20*pow((x-[7])/([8]-[7]),3)*pow(([8]-x)/([8]-[7]),3)";
    fstring += "+ [4]*15*pow((x-[7])/([8]-[7]),4)*pow(([8]-x)/([8]-[7]),2)";
    fstring += "+ [5]*6*pow((x-[7])/([8]-[7]),5)*pow(([8]-x)/([8]-[7]),1)";
    fstring += "+ [6]*1*pow((x-[7])/([8]-[7]),6)*pow(([8]-x)/([8]-[7]),0)";
    
    if(backgroundfunc=="bern6w") fstring = "(x<120)*(" + fstring + ") + (x>130)*(" + fstring + ")";
    backgroundfunction = new TF1("Bernstein6",fstring,xmin,xmax);
    backgroundfunction->SetParameters(1,0.1,0.01,0.001,0.0001,0.00001,0.000001);
    backgroundfunction->SetParNames("c0","c1","c2","c3","c4","c5","c6","xmin","xmax");
    backgroundfunction->FixParameter(7,minmass);
    backgroundfunction->FixParameter(8,maxmass);
  }
  else if(backgroundfunc=="bern7" || backgroundfunc=="bern7w"){
    TString fstring= "[0]*1*pow((x-[8])/([9]-[8]),0)*pow(([9]-x)/([9]-[8]),7)";
    fstring += "+ [1]*7*pow((x-[8])/([9]-[8]),1)*pow(([9]-x)/([9]-[8]),6)";
    fstring += "+ [2]*21*pow((x-[8])/([9]-[8]),2)*pow(([9]-x)/([9]-[8]),5)";
    fstring += "+ [3]*35*pow((x-[8])/([9]-[8]),3)*pow(([9]-x)/([9]-[8]),4)";
    fstring += "+ [4]*35*pow((x-[8])/([9]-[8]),4)*pow(([9]-x)/([9]-[8]),3)";
    fstring += "+ [5]*21*pow((x-[8])/([9]-[8]),5)*pow(([9]-x)/([9]-[8]),2)";
    fstring += "+ [6]*7*pow((x-[8])/([9]-[8]),6)*pow(([9]-x)/([9]-[8]),1)";
    fstring += "+ [7]*1*pow((x-[8])/([9]-[8]),7)*pow(([9]-x)/([9]-[8]),0)";
    
    if(backgroundfunc=="bern7w") fstring = "(x<120)*(" + fstring + ") + (x>130)*(" + fstring + ")";
    backgroundfunction = new TF1("Bernstein7",fstring,xmin,xmax);
    backgroundfunction->SetParameters(1,0.1,0.01,0.001,0.0001,0.1,0.1,0.1);
    backgroundfunction->SetParNames("c0","c1","c2","c3","c4","c5","c6","c7","xmin","xmax");
    backgroundfunction->FixParameter(8,minmass);
    backgroundfunction->FixParameter(9,maxmass);
  }
  else{
    cout << "Background function is left uninitialized!" << endl; return(0); // exit(EXIT_FAILURE);
  }
  
  //-----------------------------------------------------------------------------------------------------------------------------



  if(backgroundAsimovInput) {
    // Experimental code to replace the input histogram with the background-only Asimov which is the best fit to the input histogam
    fitresult = h_hist->Fit(backgroundfunction,"S","",minmass,maxmass); //chi-squared
    //fitresult = h_hist->Fit(expDist,"S","",minmass,maxmass); //chi-squared - replace it with an exponential 
    fitresult->Print();
    cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
    cout << ">>>>>>>>>>>>>>> Experimental replacement of input histogram with background fit <<<<<<<<<<<<<<<<<" << endl;
    for (Int_t i=1;i<nbins+1;i++) {
      //h_hist->SetBinContent(i,expDist->Eval(h_hist->GetBinCenter(i)));
      h_hist->SetBinContent(i,backgroundfunction->Eval(h_hist->GetBinCenter(i)));
      //h_hist->SetBinError(sqrt(h_hist->GetBinCenter(i)));
    }
    cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl << endl ;
  }


  cout << "======================================================" << endl;
  cout << "============== Fit to background function  ===========" << endl;
  backgroundfunction->SetLineColor(kRed);
  TString fitOptions = "S"; // Save fit results
  if(minos) fitOptions += "E";
  if(dologlikelihood) fitOptions += "L";
  //since "N" is not specified, you get a separate Canvas from this, but otherwise, you cannot draw the fit in the main canvas either..
  backgroundfitresult = h_hist->Fit(backgroundfunction,fitOptions,"",minmass,maxmass); 
  if(backgroundfitresult->Status()!=0) {
    cout << "======= Secondary fit to background function after failure =========" << endl;
    backgroundfitresult = h_hist->Fit(backgroundfunction,fitOptions,"",minmass,maxmass);
    if(backgroundfitresult->Status()!=0) {
      cout << "****** Background fit to raw MC histogram failed as well as recovery fit ******" << endl;
      backgroundfitresult->Print();
    }
  }
  if(backgroundfitresult->CovMatrixStatus()!=3) {
    cout << "******* Status of covariance matrix: " << backgroundfitresult->CovMatrixStatus() << endl;
  }
  backgroundfunction->GetParameters(eturnonlStartParameters);
  backgroundfitresult->PrintCovMatrix(cout);
  backgroundfitresult->Print();
  cout << "======================================================" << endl << endl;






  TString residualtitle="Residuals "+histname;
  TH1D *hdiff=new TH1D("hdiff",residualtitle,nbins,xmin,xmax); //cannot be minmass, maxmass, makes the residuals go crazy
  TH1D *hdiff2 = (TH1D *)hdiff->Clone("hdiff2");
  TH1D *hdiff1sigmaNofunc = (TH1D *)hdiff->Clone("hdiff1sigmaNofunc");
  TH1D *hdiff2sigmaNofunc = (TH1D *)hdiff->Clone("hdiffsigmaNofunc");
  
  TString title = "";
  if(filename.Contains("diphox"))
    title += "Diphox, ";
  else if(filename.Contains("sherpa"))
    title += "Sherpa, ";  
  else if(filename.Contains("resbos"))
    title += "Resbos, ";
  title += backgroundfunction->GetName();
  title += ", CP"; 
  title += category;
  title += "   ";

  h_hist->GetYaxis()->SetTitle(title);


  // Display the residuals with respect to the fit and the fit confidence intervals
  TCanvas *diffc=new TCanvas("diffc","Difference canvas");
  diffc->cd();

  if(showfitbands) {
    TH1D *hdiff1sigma = (TH1D *)h_hist->Clone("hdiff1sigma");
    TH1D *hdiff2sigma = (TH1D *)h_hist->Clone("hdiff2sigma");
    (TVirtualFitter::GetFitter())->GetConfidenceIntervals(hdiff1sigma,0.6827);
    (TVirtualFitter::GetFitter())->GetConfidenceIntervals(hdiff2sigma,0.9545);
    Double_t dy;
    for(Int_t i=1;i<nbins+1;i++) {
      dy =  (h_hist->GetBinContent(i) - backgroundfunction->Eval(h_hist->GetBinCenter(i)));
      hdiff->SetBinContent(i,dy);
      hdiff2->SetBinContent(i,dy);
      hdiff2->SetBinError(i,h_hist->GetBinError(i));
      hbackgroundExpectation->SetBinContent(i, backgroundfunction->Eval(h_hist->GetBinCenter(i)));
    }

    hbackgroundExpectation->Write("bkg");

    // Set bins to zero to show bands around 0 (fit-fit)
    for(Int_t i=1;i<hdiff1sigma->GetNbinsX()+1;i++) {
      hdiff1sigmaNofunc->SetBinContent(i,0);
      hdiff2sigmaNofunc->SetBinContent(i,0);
      hdiff1sigmaNofunc->SetBinError(i,hdiff1sigma->GetBinError(i));
      hdiff2sigmaNofunc->SetBinError(i,hdiff2sigma->GetBinError(i));
    }

    hdiff1sigmaNofunc->SetFillColor(kGreen);
    hdiff2sigmaNofunc->SetFillColor(kYellow);
    hdiff2sigmaNofunc->SetMaximum(h_hist->GetMaximum());
    hdiff2sigmaNofunc->SetMinimum(-h_hist->GetMaximum());
    hdiff2sigmaNofunc->GetYaxis()->SetTitleOffset(0.4);
    hdiff2sigmaNofunc->SetTitleSize(0.09,"Y");
    hdiff2sigmaNofunc->GetYaxis()->SetTitle(title);
    hdiff2sigmaNofunc->GetXaxis()->SetRange(1,hdiff2sigmaNofunc->FindBin(159.99));
    hdiff2sigmaNofunc->Draw("E3");
    hdiff1sigmaNofunc->SetLineColor(kRed);
    hdiff1sigmaNofunc->SetLineWidth(2);
    hdiff1sigmaNofunc->Draw("HE3 same");
    hdiff2->SetMarkerStyle(20);
    hdiff2->Draw("pe same");
  } // end if(showfitbands)
  else {
    Double_t dy;
    for(Int_t i=1;i<nbins+1;i++) {
      dy =  (h_hist->GetBinContent(i) - backgroundfunction->Eval(h_hist->GetBinCenter(i)));
      hdiff->SetBinContent(i,dy);
      hdiff2->SetBinContent(i,dy);
      hdiff2->SetBinError(i,h_hist->GetBinError(i));
      hbackgroundExpectation->SetBinContent(i, backgroundfunction->Eval(h_hist->GetBinCenter(i)));
    }
    
    hbackgroundExpectation->Write("bkg");
    hdiff->SetMarkerStyle(20);
    hdiff->Draw("pe");
  }

  TCanvas *myc=new TCanvas("myc","Main canvas");

  float epsilon = 0.005;
  TPad *pad1 = new TPad("pad1", "pad1",0.01,0.5,0.99,0.99);
  pad1->SetBottomMargin(0);
  pad1->Draw();
  pad1->cd();
  //eraselabel(pad1,h_hist->GetXaxis()->GetLabelSize()); //doesn't seem to do any good

  // Draw histogram and fitted function
  h_hist->GetYaxis()->SetTitleOffset(0.4);
  h_hist->SetTitleSize(0.09,"Y");
  h_hist->SetMarkerStyle(20);
  h_hist->SetMarkerSize(0.8);
  if(showfitbands) {
    TH1D *hband1sigma = (TH1D *)h_hist->Clone("hband1sigma");
    TH1D *hband2sigma = (TH1D *)h_hist->Clone("hband2sigma");
    (TVirtualFitter::GetFitter())->GetConfidenceIntervals(hband1sigma,0.6827);
    (TVirtualFitter::GetFitter())->GetConfidenceIntervals(hband2sigma,0.9545);
    hband1sigma->SetFillColor(kGreen);
    hband2sigma->SetFillColor(kYellow);
    hband2sigma->SetMaximum(h_hist->GetMaximum());
    hband2sigma->SetMinimum(0);
    hband2sigma->GetYaxis()->SetTitleOffset(0.4);
    hband2sigma->SetTitleSize(0.09,"Y");
    hband2sigma->GetYaxis()->SetTitle(title);
    hband1sigma->SetMarkerStyle(0);
    hband2sigma->SetMarkerStyle(0);
    hband1sigma->SetMarkerSize(0);
    hband2sigma->SetMarkerSize(0);
    hband2sigma->Draw("E4 X0");
    hband1sigma->Draw("E4 X0 same");
    h_hist->Draw("pe same");
  }
  else {
    h_hist->Draw("pe");
  }

  myc->cd(); //really necessary - otherwise, you are drawing this pad inside the previous pad
  
  TPad *pad2 = new TPad("pad2", "pad2",0.01,0.01,0.99,0.5+epsilon);
  pad2->SetTopMargin(0); 
  pad2->Draw();
  pad2->cd();




  // ------------  Spurious signal studies --------------

  TF1 *spurious; //to be initialized in the loops


  // Gaussian implementation of spurious signal

  bool useGaussianSpurious = 0;
  TString spuriousfunc = "";
  Double_t FWHM=0;
  if(useGaussianSpurious){
    spuriousfunc += binwidth;
    spuriousfunc += "*[0]*exp(-pow((x-[1])/[2],2)/2)/(sqrt(2*3.1415)*[2])";
    
    //Find the correct FWHM for the category in question:
    if(histname.Contains("1"))
      FWHM = 3.4;
    else if(histname.Contains("2"))
      FWHM = 3.3;
    else if(histname.Contains("3"))
      FWHM = 4.0;
    else if(histname.Contains("4"))
      FWHM = 3.9;
    else if(histname.Contains("5"))
      FWHM = 3.9;
    else if(histname.Contains("6"))
      FWHM = 3.6;
    else if(histname.Contains("7"))
      FWHM = 4.7;
    else if(histname.Contains("8"))
      FWHM = 4.5;
    else if(histname.Contains("9"))
      FWHM = 5.9;
  }
  //FWHM/=3;
  //cout << "FWHW now: " << FWHM;

  //Define parameters for the Crystal Ball function
  double alphaCB;
  double sigmaG;
  double meanG;
  double fractionCB;
  double sigmaCB;
  double meanCB;
  
  TString modelfunc = backgroundfunction->GetName();
  modelfunc += "+signalmodel";
  //cout << "setting up Model: " << modelfunc << endl;
  TF1 *model;
  int number_of_free_background_parameters; //found after fixing parameters (like range for Bernstein polynomials)
  //int number_of_background_parameters = backgroundfunction->GetNumberFreeParameters(); //this is the parameter place for the spurious amplitude 
  int number_of_background_parameters = backgroundfunction->GetNpar(); //this is the parameter place for the spurious amplitude 
  //backgroundfunction->Print();
  




  //Scan over the masses to find the spurious signal 
  //from a fit to the histogram
  //and 
  //from a fit to the Asimov pseudo data based on the histogram 
  // - estimate the error on the spurious signal 

  Double_t mass;
  Double_t chisquared;
  Double_t amplitude, amperror;
  TH1D *hamplitude = new TH1D("hamplitude","Spurious amplitude vs spurious mass",nbins,xmin,xmax);
  TH1D *hchisquared = new TH1D("hchisquared","Chi-squared vs spurious mass",nbins,xmin,xmax);


  //to get the mass with the maximum, minimimum and no bias for the MC toys
  float massMaxSpurious  = 0;
  float massMinSpurious  = 0;
  float massZeroSpurious = 0;
  float smallestDiscrepancyFromZero = 10; //random number that hopefully will be bigger than the spurious amplitude closest to zero

  //to store the maximum and minimum of spurious amplitude in different intervals
  float maxSpurious_110150         = 0;
  float minSpurious_110150         = 0;
  float maxSpurious_120130         = -100; //random number that hopefully will be smaller than the biggest spurious amplitude in 120-130
  float maxAbsoluteSpurious_110150 = 0;
  int bin120 = h_hist->FindBin(120);
  int bin130 = h_hist->FindBin(130);
  
  //the largest N_SP/sigma0 (at largest percent of background uncertainty)
  float maxsignificanceBiasAsimov = 0; 
  float mass_maxsignificanceBiasAsimov = 0;

  // Make an Asimov dataset
  TH1D *asimov = (TH1D *)h_hist->Clone();
  for (Int_t i=1;i<nbins+1;i++) {
    asimov->SetBinError(i,sqrt(asimov->GetBinContent(i)));
  }
  
  
  TH1D *hAamplitude = new TH1D("hAamplitude","Asimov spurious amplitude vs spurious mass",nbins,xmin,xmax);
  TH1D *hbgtest = new TH1D("hbgtest","Asimov spurious amplitude/sigma_muhat vs spurious mass",nbins,xmin,xmax);
  TH1D *hbandB20 = new TH1D("hbandB20","Unbiased band 20%*sigma muhat for Asimov",nbins,xmin,xmax);
  //TH1D *hbandS10 = new TH1D("hbandS10","Unbiased band +-10%*signal",nbins,xmin,xmax);

  float maxAsmiovError = 0; //for the y-axis of the plot


  //We want the fit results between 110 and 150
  float startMassFit = 110.;
  float stopMassFit = 150.;
  int startBinFit = h_hist->FindBin(startMassFit)-1;
  int stopBinFit = h_hist->FindBin(stopMassFit)+1;
  //check that the bins do not stretch longer than 150 - 110, if so, move one bin 
  if(h_hist->GetBinCenter(startBinFit)-binwidth/2 < startMassFit) startBinFit++;  
  if(h_hist->GetBinCenter(stopBinFit)+binwidth/2 > stopMassFit) stopBinFit--;  

  // Set fit options for MC and Asimov
  fitOptions += "N";
  if(!verbose) {
    fitOptions += "Q";
  }
  cout << "Fit options: " << fitOptions << endl;


  //------- The scan  
  for(Int_t imass=startBinFit; imass<=stopBinFit ;imass+=1) {

    if(verbose) 
      cout << endl << "**********  In the scan, MC and Asimov - fitting background+spurious signal  **********" << endl;
    
    mass = hamplitude->GetBinCenter(imass);

    // if(spurious!=NULL){ delete spurious; } //avoid leaking memory, but keep the last one
    cout << "spurious" << spurious << endl;
    if(useGaussianSpurious)
      spurious = new TF1("signalmodel",spuriousfunc,minmass,maxmass);
    else
      spurious = get_signalmodel(category, mass, binwidth, alphaCB, sigmaG, meanG, fractionCB, sigmaCB, meanCB);    
    //spurious = get_signalmodel(category, mass, binwidth);
    

    //if(model!=NULL) { delete model; } //avoid leaking memory, but keep the last one
    cout << "model: " << model << endl;
    //backgroundfunction->Print();
    model = new TF1("model",modelfunc,minmass,maxmass);
    //model->Print();
    //cout << "all should be set up now!" << endl;
    
    // Background specific tuning of start parameters and limits in order to improve convergence properties
    if(backgroundfunc=="eturnonl" || backgroundfunc=="expturnon") {
      model->SetParLimits(2,0,0.2); // Limits so that turn-on form is kept
      model->SetParLimits(3,0.05,1);
    }
    else if(backgroundfunc=="exp" || backgroundfunc=="expw") {
      model->SetParameter(0,h_hist->GetSumOfWeights()); 
      model->SetParLimits(1,1E-4,0.2);
    }
    else if(backgroundfunc=="bern3" || backgroundfunc=="bern3w") {
      model->FixParameter(4,minmass);
      model->FixParameter(5,maxmass);
    }
    else if(backgroundfunc=="bern4" || backgroundfunc=="bern4w") {
      model->FixParameter(5,minmass);
      model->FixParameter(6,maxmass);
    }
    else if(backgroundfunc=="bern5" || backgroundfunc=="bern5w") {
      model->FixParameter(6,minmass);
      model->FixParameter(7,maxmass);
    }
    else if(backgroundfunc=="bern6" || backgroundfunc=="bern6w") {
      model->FixParameter(7,minmass);
      model->FixParameter(8,maxmass);
    }
    else if(backgroundfunc=="bern7" || backgroundfunc=="bern7w") {
      model->FixParameter(8,minmass);
      model->FixParameter(9,maxmass);
    }
    
    
    number_of_free_background_parameters = backgroundfunction->GetNumberFreeParameters(); //this is the parameter place for the spurious amplitude 

     
    if(useGaussianSpurious){
      //spuriousfunc += "*[0]*exp(-pow((x-[1])/[2],2)/2)/(sqrt(2*3.1415)*[2])";
      model->FixParameter(number_of_background_parameters+1,mass); 
      model->FixParameter(number_of_background_parameters+2,FWHM/2.35482); 
    } else {
      FixParametersOfSpuriousModel(number_of_background_parameters, model, binwidth, alphaCB, sigmaG, meanG, fractionCB, sigmaCB, meanCB);
    }
    // model->Print();




    if(verbose) cout << endl << "**********  Fit to MC  **********" << endl;

    if(turnOffSpurious)
	{model->FixParameter(number_of_background_parameters,0);}
    else
      {
	model->ReleaseParameter(model->GetParNumber("shat"));
	model->SetParameter(model->GetParNumber("shat"),0);
      }
							    
							     
    int fitstatus = -1;
							     cout << " What???? " << endl;
							     h_hist->Print();
										     h_hist->Draw();
								     h_hist->SetTitle("say what?");

								     
								     model->Print();
								     fitresult = h_hist->Fit(model,fitOptions,"",minmass,maxmass);
								     
    fitstatus = fitresult->Status();
    if(fitstatus!=0) {
      cout << "****** Failed fit to MC at mass " << mass << endl;
      failedFitsMC += 1;
    }
    if(fitresult->CovMatrixStatus()!=3) {
      cout << "******* In fit to MC: Status of covariance matrix: " << fitresult->CovMatrixStatus() << endl;
      h_hist->Draw();
      fitresult->Print();
      fitresult->PrintCovMatrix(cout);
      errorMatrixProblemsMC += 1;
    }
    chisquared = model->GetChisquare();
    amplitude = model->GetParameter(number_of_background_parameters);
    amperror = model->GetParError(number_of_background_parameters);
    if(verbose){
      cout << "************** mass = " << mass << " GeV " 
	   << ", chi-squared=" << chisquared << ", spurious signal=" << amplitude << endl;
    }

    //only fill info if the fit was sucessfull
    if(fitstatus==0) {
      hamplitude->SetBinContent(imass,amplitude);
      hamplitude->SetBinError(imass,amperror);
      hchisquared->SetBinContent(imass,chisquared);
    }

    if(h_hist->GetXaxis()->GetBinCenter(imass) >= 110 && h_hist->GetXaxis()->GetBinCenter(imass) <= 150 ) {
      if(amplitude>maxSpurious_110150){ 
	maxSpurious_110150=amplitude;
	massMaxSpurious  = h_hist->GetXaxis()->GetBinCenter(imass);
      }
      if(amplitude<minSpurious_110150) { 
	minSpurious_110150=amplitude;
	massMinSpurious  = h_hist->GetXaxis()->GetBinCenter(imass);
      }
      if( (imass>=bin120) && (imass<bin130) ){
	if(amplitude>maxSpurious_120130) maxSpurious_120130=amplitude;
      }
      if(fabs(amplitude)<smallestDiscrepancyFromZero){
	smallestDiscrepancyFromZero =  fabs(amplitude);
	massZeroSpurious = h_hist->GetXaxis()->GetBinCenter(imass);
      }    
    }    

    
    //Now the Asimov fit

    if(verbose) 
      cout << endl << "**********  Fit to Asimov  **********" << endl;
    

    if(turnOffSpurious)
      model->FixParameter(number_of_background_parameters,0);
    else
      model->ReleaseParameter(model->GetParNumber("shat"));
      model->SetParameter(model->GetParNumber("shat"),0); //make Minuit work a bit (otherwise, it will probably be the same as for the histogram fit)
    

    fitresult = asimov->Fit("model",fitOptions,"",minmass,maxmass);
    fitstatus = fitresult->Status();
    if(fitstatus!=0) {
      cout << "****** Failed fit to Asimov at mass " << mass << endl;
      fitresult->Print();
      failedFitsAsimov += 1;
    }
    if(fitresult->CovMatrixStatus()!=3) {
      cout << "******* In fit to Asimov: Status of covariance matrix: " << fitresult->CovMatrixStatus() << endl;
      errorMatrixProblemsAsimov += 1;
    }

    amplitude = model->GetParameter(number_of_background_parameters);
    amperror = model->GetParError(number_of_background_parameters);
    if(verbose) {
      cout << endl << "**** Asimov fit " << endl;
      cout << "************* mass = " << mass << " GeV ";
      cout << ", spurious signal = " << amplitude << endl;
    }
    
    //only fill info if the fit was sucessfull
    if(fitstatus==0) {
      hAamplitude->SetBinContent(imass,amplitude);
      hAamplitude->SetBinError(imass,amperror);
      hbgtest->SetBinContent(imass,amplitude/amperror);
      hbandB20->SetBinError(imass,0.20*amperror);
      // hbandS10->SetBinError(imass,signal*0.10);

     
      float significanceBias = amplitude/amperror;
      if(fabs(significanceBias)>fabs(maxsignificanceBiasAsimov)) {
	maxsignificanceBiasAsimov = significanceBias;
	mass_maxsignificanceBiasAsimov = mass;
      }


      //Fill expected and observed p0 and significance

      float expected_signal = hexpectedSignal->GetBinContent(hexpectedSignal->FindBin(mass));
      float expected_unbiased_significance = expected_signal/amperror;
      float expected_p0_asimov =  Pvalue_uncapped(expected_unbiased_significance);
      float observed_significance = amplitude/amperror;
      float observed_p0_asimov =  Pvalue_uncapped(observed_significance);

      hp0_expected_asimov->SetBinContent(imass,expected_p0_asimov);
      hp0_observed_asimov->SetBinContent(imass,observed_p0_asimov);

      hsignificance_expected_asimov->SetBinContent(imass,expected_unbiased_significance);
      hsignificance_observed_asimov->SetBinContent(imass,observed_significance);


    }

    if(fabs(amperror)>maxAsmiovError) maxAsmiovError = amperror;

   
  }

  //Find the largest absolute value : 
  (maxSpurious_110150 > fabs(minSpurious_110150)) ? maxAbsoluteSpurious_110150=maxSpurious_110150 : maxAbsoluteSpurious_110150=fabs(minSpurious_110150) ; 
  float massMaxAbsoluteSpurious  = 0;
  (maxSpurious_110150 > fabs(minSpurious_110150)) ? massMaxAbsoluteSpurious = massMaxSpurious : massMaxAbsoluteSpurious = massMinSpurious ;





  // Throw some toys, fit to spurious signal at a given mass, histogram amplitude.
  // We are going to do this for 3 masses corresponding to ~zero bias, maximum positive bias and maxmimum negative bias:
  // massZeroSpurious, massMaxSpurious, massMinSpurious

  if(backgroundAsimovInput || turnOffSpurious) {
    massMinSpurious = 110;
    massMaxSpurious = 125;
    massZeroSpurious = 140;
    massMaxAbsoluteSpurious = 125;
  }

  Double_t toymass[3]={massMinSpurious, massMaxSpurious, massZeroSpurious};

  htoys=(TH1D *)h_hist->Clone();
  
  TH1D *htoysshatMax = new TH1D("htoysshatMax","shat dist. for maximum bias",100,-200,200);  
  TH1D *htoysshatMin = new TH1D("htoysshatMin","shat dist. for minimum bias",100,-200,200);  
  TH1D *htoysshatZero = new TH1D("htoysshatZero","shat distibution ~zero bias",100,-200,200);
  TH1D *hchisquareMax = new TH1D("hchisquareMax","Chisquared dist. for max. bias",100,0,250);
  TH1D *hchisquareMin = new TH1D("hchisquareMin","Chisquared dist. for max. bias",100,0,250);
  TH1D *hchisquareZero = new TH1D("hchisquareZero","Chisquared dist. for max. bias",100,0,250);

  // Pointers to be able to refer to histograms in the loop
  TH1D* hshat[3];
  TH1D* hchisq[3];
  hshat[0] = htoysshatMax;
  hshat[1] = htoysshatMin;
  hshat[2] = htoysshatZero;
  hchisq[0] = hchisquareMax;
  hchisq[1] = hchisquareMin;
  hchisq[2] = hchisquareZero;

  // Start parameters for first toy. After the first, use the results of the previous toy
  // as the start for the next - makes the next fit fast and reliable.
  // model->SetParameter(0,3000);
  // model->SetParameter(1,-0.025);
  // model->SetParameter(3,0.0);

  Double_t shat[3];
  Double_t dshat[3];
  Double_t dmass[3]={0,0,0};
  Double_t RMSshat[3];


  //Pulls of toys
  TH1F* hist_pulls_pearson_toys = new TH1F("hist_pulls_pearson_toys","",numbins100160,-5,5);
  TH1F* hist_pulls_squared_pearson_toys = new TH1F("hist_pulls_squared_pearson_toys","",numbins100160,0,10);
  float range_sum_pulls_pearson_toys = 15;
  TH1F* hist_sum_pulls_pearson_toys = new TH1F("hist_sum_pulls_pearson_toys","",numbins100160,-range_sum_pulls_pearson_toys,range_sum_pulls_pearson_toys);
  TH1F* hist_sum_pulls_squared_pearson_toys = new TH1F("hist_sum_pulls_squared_pearson_toys","",numbins100160,50,150);
  float sum_of_pearson_toy_pulls = 0.;          //sum of the pulls of all bins for one toy
  float sum_of_pearson_toy_pulls_squared = 0.;  //sum of the square of the pulls of all bins for one toy


  if(backgroundfunc=="eturnonl") backgroundfunction->SetParameters(eturnonlStartParameters);

  // Loop over the 3 test masses for the toys
  for(Int_t itoymass=0;itoymass<3;itoymass++) {
    hshat[itoymass]->Sumw2(); // Calculate unbinned mean and RMS

    //set up model
    if(spurious!=NULL){ delete spurious; } //avoid leaking memory, but keep the last one
    if(useGaussianSpurious)
      spurious = new TF1("signalmodel",spuriousfunc,minmass,maxmass);
    else
      spurious = get_signalmodel(category, toymass[itoymass], binwidth, alphaCB, sigmaG, meanG, fractionCB, sigmaCB, meanCB);    
    //spurious = get_signalmodel(category, mass, binwidth);

    if(model!=NULL) { delete model; } //avoid leaking memory, but keep the last one
    model = new TF1("model",modelfunc,minmass,maxmass);

    // Background specific tuning of start parameters and limits in order to improve convergence properties
    if(backgroundfunc=="eturnonl" || backgroundfunc=="expturnon") {
      model->SetParLimits(2,0,0.2); // Limits so that turn-on form is kept
      model->SetParLimits(3,0.05,1);
    }
    else if(backgroundfunc=="exp" || backgroundfunc=="expw") {
      model->SetParameter(0,htoys->GetSumOfWeights()); 
      model->SetParameter(1,betaguess);
      model->SetParLimits(1,0.0,1.0);
    }
    else if(backgroundfunc=="doubleexp") {
      model->SetParameter(0,htoys->GetSumOfWeights()); 
      // model->SetParameter(1,betaguess);
      model->SetParameter(1,betaguess/10);
      model->SetParLimits(1,0.0,1.0);
      model->SetParLimits(2,0.0,htoys->GetSumOfWeights());
      model->SetParameter(2,htoys->GetSumOfWeights()/20);
      model->SetParameter(3,betaguess*3);
      model->SetParLimits(3,0.0,1.0);
    }
    else if(backgroundfunc=="bern3" || backgroundfunc=="bern3w") {
      model->FixParameter(4,minmass);
      model->FixParameter(5,maxmass);
    }
    else if(backgroundfunc=="bern4" || backgroundfunc=="bern4w") {
      model->FixParameter(5,minmass);
      model->FixParameter(6,maxmass);
    }
    else if(backgroundfunc=="bern5" || backgroundfunc=="bern5w") {
      model->FixParameter(6,minmass);
      model->FixParameter(7,maxmass);
    }
    else if(backgroundfunc=="bern6" || backgroundfunc=="bern6w") {
      model->FixParameter(7,minmass);
      model->FixParameter(8,maxmass);
    }
    else if(backgroundfunc=="bern7" || backgroundfunc=="bern7w") {
      model->FixParameter(8,minmass);
      model->FixParameter(9,maxmass);
    }
     

    number_of_free_background_parameters = backgroundfunction->GetNumberFreeParameters(); //this is the parameter place for the spurious amplitude 

    if(useGaussianSpurious){
      //spuriousfunc += "*[0]*exp(-pow((x-[1])/[2],2)/2)/(sqrt(2*3.1415)*[2])";
      model->FixParameter(number_of_background_parameters+1,toymass[itoymass]); 
      model->FixParameter(number_of_background_parameters+2,FWHM/2.35482); 
    } else {
      FixParametersOfSpuriousModel(number_of_background_parameters, model, binwidth, alphaCB, sigmaG, meanG, fractionCB, sigmaCB, meanCB);
    }

    //int fitstatus = -1;
    double precision;

    // Special fit options setup for toys, want loglikelihood by default, recovery is chi-squared
    TString fitOptionsToys = "SNL";
    TString fitOptionsToysRecovery = "SN";
    if(minos) {
      fitOptionsToys += "E";
      fitOptionsToysRecovery += "E";
    }
    if(improve) {
      fitOptionsToys += "M";
      fitOptionsToysRecovery += "M";
    }
    if(!verbose) {
      fitOptionsToys += "Q";
      fitOptionsToysRecovery += "Q";
    }



    cout << endl << ">>>>>>>>>>>>> Starting toys for mass " << toymass[itoymass] << endl; 

    for(Int_t itoy=0;itoy<number_of_toys;itoy++){

      htoys->Reset();
      htoys->FillRandom(h_hist,(Int_t)h_hist->Integral());


      
      if(turnOffSpurious){
	//Fix spurious signal to zero 
	model->FixParameter(number_of_background_parameters,0);
      }
      else{
	cout << "******* " << model->GetParameter("shat") << endl;
	// Start each toy with zero spurious amplitude
	model->ReleaseParameter(model->GetParNumber("shat"));
	model->SetParameter(model->GetParNumber("shat"),0); 
      }

      fitresult = htoys->Fit("model",fitOptionsToys,"",minmass,maxmass); 
      if(fitresult->Status()!=0) {
      	 cout << "****** Failed fit for toy " << itoy << " at mass " << toymass[itoymass] << " GeV " << endl;
      	 fitresult->Print();
	fitresult = htoys->Fit("model",fitOptionsToysRecovery,"",minmass,maxmass); // !!! Recovery is chi-squared fit (for small signal fit failures)
	if(fitresult->Status()!=0) cout << "****** Failed to recover failed fit with chi-squared method" << endl;
      	// htoys->Draw();
	// model->Draw("same");
      	// return(0);
      } 
      if(fitresult->Status()==0) {
	hshat[itoymass]->Fill(model->GetParameter(number_of_background_parameters));
	hchisq[itoymass]->Fill(model->GetChisquare());
      } else {
	failedFitsToys += 1;
      }
      if(fitresult->CovMatrixStatus()!=3) {
	cout << "******* In toy fit: Status of covariance matrix: " << fitresult->CovMatrixStatus() << endl;
	fitresult->Print();
	fitresult->PrintCovMatrix(cout);
	errorMatrixProblemsToys += 1;
	// htoys->Draw();
	// model->Draw("same");
	// return(0);
      }
      
      
      //Pulls
      int bin100GeV = htoys->FindBin(100);
      int bin160GeV = htoys->FindBin(160);
      //check that the bins do not stretch longer than 100 - 160, if so, move one bin and give a warning if that cuts the mass range
      if(htoys->GetBinCenter(bin100GeV)-binwidth/2 < 100) { bin100GeV++;  
	if(htoys->GetBinCenter(bin100GeV)-binwidth/2 > 100) 
	  cout << "OBS: For the pulls in toys, you start from " << htoys->GetBinCenter(bin100GeV)-binwidth/2 << "GeV!" << endl; }
      if(htoys->GetBinCenter(bin160GeV)+binwidth/2 > 160) { bin160GeV--;  
      	if(htoys->GetBinCenter(bin160GeV)+binwidth/2 < 160)
	  cout << "OBS: For the pulls in toys, you stop at " << htoys->GetBinCenter(bin160GeV)+binwidth/2 << "GeV!" << endl; }
      
      for(int bin=bin100GeV; bin<bin160GeV+1; bin++){
	float observation = htoys->GetBinContent(bin);
	float expectation = model->Eval(htoys->GetBinCenter(bin));
	float pearsonToyPull = (observation-expectation)/sqrt(expectation);
	sum_of_pearson_toy_pulls += pearsonToyPull;
	float pearsonSquaredToyPull = pow(observation-expectation,2)/expectation;
	sum_of_pearson_toy_pulls_squared += pearsonSquaredToyPull;

	hist_pulls_pearson_toys->Fill(pearsonToyPull);
	hist_pulls_squared_pearson_toys->Fill(pearsonSquaredToyPull);

      }
      hist_sum_pulls_pearson_toys->Fill(sum_of_pearson_toy_pulls);
      hist_sum_pulls_squared_pearson_toys->Fill(sum_of_pearson_toy_pulls_squared);
      sum_of_pearson_toy_pulls = 0.;
      sum_of_pearson_toy_pulls_squared = 0.;

    }

    shat[itoymass] = hshat[itoymass]->GetMean();
    dshat[itoymass] = hshat[itoymass]->GetRMS()/sqrt(hshat[itoymass]->GetEntries());
    RMSshat[itoymass] = hshat[itoymass]->GetRMS();
    // Print the mean, error and sigma for these toys
    cout << "shat(" << toymass[itoymass] << ") = " << shat[itoymass]
  	 << " +- " << dshat[itoymass]
  	 << ", RMS = " << RMSshat[itoymass] << endl;


  }
  // Make a TGraph of the results, thick error bars for the uncertainty, narrow for the RMS
  TGraphErrors *tgerror=new TGraphErrors(3,toymass,shat,dmass,dshat);
  TGraphErrors *tgRMS=new TGraphErrors(3,toymass,shat,dmass,RMSshat);




  //--------  Draw the results in the main canvas

  //Draw the Asimov results
  hAamplitude->SetLineColor(kBlack);
  hAamplitude->SetMarkerStyle(20);
  hAamplitude->SetMarkerSize(0.4);
  hAamplitude->SetFillColor(920); //892 mørkrosa, 920 lysegrå, 900 rød
  hAamplitude->Draw("PE4");
  hbandB20->SetFillColor(414); //833 dusere grønn
  hbandB20->Draw("sameE4");
  // Ugh, need to redraw points, but with no error bars to get the black points on top of the green lines, how to do this elegantly?!
  // Double ugh, according to documentation, draw option P is not supposed to draw empty bins!!!
  TH1D *tmp=(TH1D*)hAamplitude->Clone();
  tmp->SetAxisRange(startMassFit,stopMassFit,"X"); //Not to draw the points along zero outside the fitted region 
  for(int imass=0;imass<nbins;imass++) {
    tmp->SetBinError(imass,0);
  }
  tmp->Draw("Psame");

  //  Draw the residuals, the residuals integrated over 10 GeV and the spurious fit amplitude
  if(!backgroundAsimovInput) hdiff->Draw("same");
  hdiff->GetXaxis()->SetTitle("M#gamma#gamma [GeV]");
  TH1 *hdiffClone = (TH1 *)hdiff->Clone("hdiffClone");
  hdiffClone->SetLineColor(kRed);
  hdiffClone->SetLineWidth(2);
  if(!backgroundAsimovInput) hdiffClone->Draw("same");
  int rebinfactor_10GeV = (int)(10/((xmax-xmin)/nbins)); //to get 10 GeV per bin
  hdiffClone->Rebin(rebinfactor_10GeV);
  //hamplitude->SetLineStyle(kDashed);
  hamplitude->SetLineWidth(2);
  hamplitude->SetMarkerStyle(20);
  hamplitude->SetMarkerSize(0.8); 
  hamplitude->SetMarkerColor(603);
  hamplitude->SetLineColor(603);
  hamplitude->Draw("same");


  //To find the minimum within the mass range (to take into account all three histograms, only in the desired mass range)  
  float ymindiff = 0;
  float ymaxdiff = 0;
  //float amplitudeerror = 0; //if the error on the amplitude is much bigger than ymindiff or ymaxdiff (not the case for MC)

  //hdiff
  for (int bin =  hdiff->FindBin(minmass); bin < hdiff->FindBin(maxmass)+1; bin++){
    if(hdiff->GetBinContent(bin)<ymindiff) ymindiff = hdiff->GetBinContent(bin);
    if(hdiff->GetBinContent(bin)>ymaxdiff) ymaxdiff = hdiff->GetBinContent(bin);
  }
  //hdiffClone:
  for (int bin = hdiffClone->FindBin(minmass); bin < hdiffClone->FindBin(maxmass)+1; bin++){
    if(hdiffClone->GetBinContent(bin)<ymindiff) ymindiff = hdiffClone->GetBinContent(bin);
    if(hdiffClone->GetBinContent(bin)>ymaxdiff) ymaxdiff = hdiffClone->GetBinContent(bin);
  }
  //hamplitude:
  for (int bin = hamplitude->FindBin(minmass); bin < hamplitude->FindBin(maxmass)+1; bin++){
    if(hamplitude->GetBinContent(bin)<ymindiff) ymindiff = hamplitude->GetBinContent(bin);
    if(hamplitude->GetBinContent(bin)>ymaxdiff) ymaxdiff = hamplitude->GetBinContent(bin);
    //if(hamplitude->GetBinError(bin)>amplitudeerror) amplitudeerror = hamplitude->GetBinError(bin);
  }

  //Finally, set the reasonable y-axis range
  //hdiff->SetAxisRange(ymindiff-amplitudeerror,ymaxdiff+amplitudeerror,"Y");
  //hdiff->SetAxisRange(ymindiff+ymindiff*0.2,ymaxdiff+ymaxdiff*0.2,"Y");
  //hdiff->SetAxisRange(ymindiff-maxAsmiovError,ymaxdiff+maxAsmiovError,"Y");
  hdiff->SetAxisRange(ymindiff-maxAsmiovError*1.2,ymaxdiff+maxAsmiovError*1.2,"Y");


  //Line at zero
  TLine *tl=new TLine(minmass,0,maxmass,0);
  tl->Draw("same");


  // Draw the toy results
  if(number_of_toys>0) {
    tgerror->SetLineColor(kRed);
    tgerror->SetLineWidth(4);
    tgerror->SetMarkerStyle(20);
    tgRMS->SetLineColor(kBlue);
    tgRMS->SetLineWidth(1);
    tgRMS->SetMarkerStyle(0);

    tgerror->Draw("Psame");
    tgRMS->Draw("Psame");
  }


  // Draw the lines which indicate the +-10% (of the expected SM signal) 
  // limits of the size of the spurious signal
  TH1D *signalLimitUpper = new TH1D("signalLimitUpper","Signal rate vs mH",nbins,xmin,xmax);
  TH1D *signalLimitLower = new TH1D("signalLimitLower","Signal rate vs mH",nbins,xmin,xmax);
  //signalRate(category,signalLimitUpper,luminosity);
  //signalRate(category,signalLimitLower,luminosity);
  signalLimitUpper->Scale(0.10);
  signalLimitLower->Scale(-0.10);
  signalLimitUpper->SetLineWidth(2);
  signalLimitLower->SetLineWidth(2);
  signalLimitUpper->SetLineColor(420); //kGreen+3
  signalLimitLower->SetLineColor(420);
  signalLimitUpper->Draw("same");
  signalLimitLower->Draw("same");

  pad1->cd();


  //Legend:
  TLegend *leg = new TLegend(0.6,0.4,0.9,0.9);
  leg->SetBorderSize(1);
  leg->SetTextFont(42);
  leg->SetLineColor(1);
  leg->SetLineStyle(1);
  leg->SetLineWidth(1);
  leg->SetFillColor(0);
  leg->SetFillStyle(1001);
  leg->SetHeader("            Explanation of entries in lower pad");
  if(!backgroundAsimovInput) leg->AddEntry(hdiff,"Residual (MC-fit_{b})","l");
  if(!backgroundAsimovInput) leg->AddEntry(hdiffClone,"Residual integrated in 10 GeV","l");
  leg->AddEntry(hamplitude, "Fitted spurious to MC","lp");
  leg->AddEntry(hAamplitude, "Fitted spurious to Asimov","pf");
  if(number_of_toys>0) leg->AddEntry(tgerror, "Fitted spurious to toys (with error)","lp");
  if(number_of_toys>0) leg->AddEntry(tgRMS, "Toy B uncertainty ( #sigma_{0})","l");
  leg->AddEntry(hbandB20, "20 \% of Asimov B uncertainty ( #sigma_{0})","f");
  leg->AddEntry(signalLimitUpper, "10 \% S","l");

  leg->Draw("same");

  // Print the integrated luminosity
  char chlumi[] = "100.0";
  sprintf(chlumi,"%4.1f",luminosity);
  TString lumitext = "#int Ldt=";
  lumitext += chlumi;
  lumitext += " fb^{-1}";
  TLatex *tllumitext = new TLatex(120,0.85*h_hist->GetMaximum(),lumitext);
  tllumitext->Draw();



  //-- Canvas without the origin histogram

  TCanvas *canv_bias=new TCanvas("canv_bias","Bias canvas");
  canv_bias->cd();

  //Draw the Asimov result
  TH1 *hAamplitude_clone = (TH1*)hAamplitude->Clone("hAamplitude_clone"); //want a different range than in the origin plot


  hAamplitude_clone->SetMarkerStyle(20);
  hAamplitude_clone->SetMarkerSize(0.4);
  hAamplitude_clone->SetFillColor(920); //892 mørkrosa, 920 lysegrå, 900 rød
  hAamplitude_clone->Draw("PE4");
  hAamplitude_clone->SetAxisRange(110,150,"X");
  hAamplitude_clone->SetAxisRange(ymindiff-maxAsmiovError*1.2,ymaxdiff+maxAsmiovError*2.5,"Y");
  if(backgroundAsimovInput)
    hAamplitude_clone->SetAxisRange(-7,7,"Y");
  hAamplitude_clone->GetXaxis()->SetTitle("M#gamma#gamma [GeV]");
  Double_t tsizefrac = tsize/1.4;
  hAamplitude_clone->SetLabelFont(font,"x");
  hAamplitude_clone->SetTitleFont(font,"x");
  hAamplitude_clone->SetLabelFont(font,"y");
  hAamplitude_clone->SetTitleFont(font,"y");
  hAamplitude_clone->SetLabelSize(tsizefrac,"x");
  hAamplitude_clone->SetTitleSize(tsizefrac,"x");
  hAamplitude_clone->SetLabelSize(tsizefrac,"y");
  hAamplitude_clone->SetTitleSize(tsize,"y");
  hAamplitude_clone->GetYaxis()->SetTitleOffset(0.8);
  hAamplitude_clone->GetXaxis()->SetTitleOffset(1.2);
  hAamplitude_clone->GetYaxis()->SetTitle(title);

  hbandB20->Draw("sameE4");
  tmp->Draw("Psame");

  //  Draw the residuals, the residuals integrated over 10 GeV and the spurious fit amplitude
  if(!backgroundAsimovInput) hdiff->Draw("same");
  hdiffClone = (TH1 *)hdiff->Clone("hdiffClone");
  hdiffClone->SetLineColor(kRed);
  hdiffClone->SetLineWidth(2);
  if(!backgroundAsimovInput) hdiffClone->Draw("same");
  hdiffClone->Rebin(rebinfactor_10GeV);
  hamplitude->Draw("same");

  TLine *zeroline = new TLine(110,0,150,0);
  zeroline->Draw("same");

  // Draw the toy results
  tgerror->Draw("Psame");
  tgRMS->Draw("Psame");

  signalLimitUpper->Draw("same");
  signalLimitLower->Draw("same");


  //Legend:
  TLegend *leg1_canv_bias = new TLegend(0.2600575,0.75,0.5402299,0.89,NULL,"brNDC");
  leg1_canv_bias->SetBorderSize(0);
  leg1_canv_bias->SetTextFont(42);
  leg1_canv_bias->SetFillColor(0);
  if(!backgroundAsimovInput) leg1_canv_bias->AddEntry(hdiff,"Residual (MC-fit_{b})","l");
  if(!backgroundAsimovInput) leg1_canv_bias->AddEntry(hdiffClone,"Residual integrated in 10 GeV","l");
  leg1_canv_bias->AddEntry(hamplitude, "Fitted spurious to MC","lp");
  leg1_canv_bias->AddEntry(hAamplitude, "Fitted spurious to Asimov","pf");

  TLegend *leg2_canv_bias = new TLegend(0.5416667,0.75,0.8965517,0.8919492,NULL,"brNDC");
  leg2_canv_bias->SetBorderSize(0);
  leg2_canv_bias->SetTextFont(42);
  leg2_canv_bias->SetFillColor(0);
  if(number_of_toys>0) leg2_canv_bias->AddEntry(tgerror, "Fitted spurious to toys (with error)","lp");
  if(number_of_toys>0) leg2_canv_bias->AddEntry(tgRMS, "Toy B uncertainty ( #sigma_{0})","l");
  leg2_canv_bias->AddEntry(hbandB20, "20 \% of Asimov B uncertainty ( #sigma_{0})","f");
  leg2_canv_bias->AddEntry(signalLimitUpper, "10 \% S","l");


  leg1_canv_bias->Draw("same");
  leg2_canv_bias->Draw("same");

  // Print the integrated luminosity
  // char *chlumi="100.0";
  // sprintf(chlumi,"%4.1f",luminosity);
  // TString lumitext = "#int Ldt=";
  // lumitext += chlumi;
  // lumitext += " fb^{-1}";
  // TLatex *tllumitext = new TLatex(143,ymaxdiff+0.3*maxAsmiovError,lumitext);
  // tllumitext->SetTextSize(0.02966102);
  // tllumitext->Draw();






  //-- Canvas with zoom of the lower pad, with less information but the errors of the spurious amplitude (is it significant?)
  TCanvas *canv_zoom=new TCanvas("canv_zoom","Zoom canvas");
  canv_zoom->cd();

  //if you don't want the orginial plot to be affected:
  //TH1 *clone_hamplitude = (TH1*)hamplitude->Clone("clone_hamplitude");
  hamplitude->Draw();
  hamplitude->SetMarkerStyle(20);
  hamplitude->SetMarkerColor(kBlue);
  hamplitude->SetMarkerSize(0.8); //this is ignored, why?

  tsizefrac = tsize/2;
  hamplitude->SetLabelFont(font,"x");
  hamplitude->SetTitleFont(font,"x");
  hamplitude->SetLabelFont(font,"y");
  hamplitude->SetTitleFont(font,"y");
  hamplitude->SetLabelSize(tsizefrac,"x");
  hamplitude->SetTitleSize(tsizefrac,"x");
  hamplitude->SetLabelSize(tsizefrac,"y");
  hamplitude->SetTitleSize(tsize/1.19,"y");
  hamplitude->GetYaxis()->SetTitleOffset(0.4);

  //must clone to have different range than in the orginal plot
  TH1 *clone_hdiff = (TH1*)hdiff->Clone("clone_hdiff");
  clone_hdiff->Draw("same");
  clone_hdiff->SetLineColor(kRed);
  clone_hdiff->SetLineWidth(2);
  clone_hdiff->Rebin(rebinfactor_10GeV);

  //Draw the Asimov results
  TH1 *clone_hAamplitude = (TH1*)hAamplitude->Clone("clone_hAamplitude");
  clone_hAamplitude->Draw("hist p same");
  
  signalLimitUpper->Draw("same");
  signalLimitLower->Draw("same");

  //y-axis 
  // float clone_ymin = clone_hdiff->GetBinContent(clone_hdiff->GetMinimumBin());
  // float clone_ymax = clone_hdiff->GetBinContent(clone_hdiff->GetMaximumBin());
  // float hamplitude_ymin = hamplitude->GetBinContent(hamplitude->GetMinimumBin());
  // float hamplitude_ymax = hamplitude->GetBinContent(hamplitude->GetMaximumBin());
  // float max, min;
  // (clone_ymin < hamplitude_ymin) ? min = clone_ymin : min = hamplitude_ymin ;
  // (clone_ymax > hamplitude_ymax) ? max = clone_ymax : max = hamplitude_ymax ;

  float hamplitude_ymin = 0;
  float hamplitude_ymax = 0;
  float amplitudeerror = 0;
  for (int bin = hamplitude->FindBin(startMassFit); bin < hamplitude->FindBin(stopMassFit)+1; bin++){
    if(hamplitude->GetBinContent(bin)<hamplitude_ymin) hamplitude_ymin = hamplitude->GetBinContent(bin);
    if(hamplitude->GetBinContent(bin)>hamplitude_ymax) hamplitude_ymax = hamplitude->GetBinContent(bin);
    if(hamplitude->GetBinError(bin)>amplitudeerror) amplitudeerror = hamplitude->GetBinError(bin);
  }

  float signal_ymax = signalLimitUpper->GetBinContent(hamplitude->GetMaximumBin());
  float signal_ymin = signalLimitLower->GetBinContent(hamplitude->GetMinimumBin());
  float signalmax = 0;
  (fabs(signal_ymin)>fabs(signal_ymax)) ? signalmax = fabs(signal_ymin) : signalmax = fabs(signal_ymax) ; 

  float max, min;
  (-signalmax < hamplitude_ymin) ? min = -signalmax : min = hamplitude_ymin ;
  (signalmax > hamplitude_ymax) ? max = signalmax : max = hamplitude_ymax ;

  // cout << "signal_ymax = " << signal_ymax << endl 
  //      << "signal_ymin = " << signal_ymin << endl
  //      << "hamplitude_ymin = " << hamplitude_ymin << endl
  //      << "hamplitude_ymax = " << hamplitude_ymax << endl;
  //hamplitude->SetAxisRange(min-amplitudeerror,max+amplitudeerror,"Y");
  //hamplitude->SetAxisRange(ymindiff,ymaxdiff,"Y");
  //hamplitude->SetAxisRange(hamplitude_ymin+hamplitude_ymin*0.2,hamplitude_ymax+hamplitude_ymax*0.2,"Y");
  hamplitude->SetAxisRange(min+min*0.2,max+max*0.2,"Y");
  hamplitude->GetYaxis()->SetTitle(title);
  hamplitude->GetYaxis()->SetTitleOffset(0.9);
  //->SetTitleSize(0.09,"Y");

  
  //x-axis 
  hamplitude->GetXaxis()->SetRangeUser(110,150); 
  hamplitude->GetXaxis()->SetTitle("M#gamma#gamma [GeV]");

  tl=new TLine(startMassFit,0,stopMassFit,0);
  tl->Draw("same"); 




  //Canvas with pulls
  TCanvas *canv_pulls = new TCanvas("canv_pulls","Canvas with the Pearson pulls for toys");
  canv_pulls->cd();

  hist_pulls_pearson_toys->GetYaxis()->SetLabelSize(0.04);
  hist_pulls_pearson_toys->GetXaxis()->SetLabelSize(0.04);
  hist_pulls_pearson_toys->GetYaxis()->SetTitle(title);
  hist_pulls_pearson_toys->GetXaxis()->SetTitle("(data-fit)/#sqrt{fit}");
  hist_pulls_pearson_toys->GetXaxis()->SetTitleOffset(0.9);
  hist_pulls_pearson_toys->SetMarkerStyle(20);
  hist_pulls_pearson_toys->Draw("pe");

  TPaveText *pt = new TPaveText(0.7,0.73,0.84,0.84,"NDC");
  pt->SetTextAlign(11); //left aligned
  pt->SetTextSize(0.03); pt->SetFillColor(0); pt->SetBorderSize(0);

  float expectedrmsNumber = sqrt((float)(numbins100160-number_of_free_background_parameters)/(float)numbins100160);
  // TString expected_rms             = "Expected RMS : ";
  // expected_rms += Form("%3.2f",expectedrmsNumber);

  TString mean_pulls_Pearson = "Mean : ";
  mean_pulls_Pearson +=  Form("%3.3f", hist_pulls_pearson_toys->GetMean());
  mean_pulls_Pearson += " #pm ";
  mean_pulls_Pearson += Form( "%3.3f", hist_pulls_pearson_toys->GetMeanError());

  TString rms_pulls_Pearson = "RMS : ";
  rms_pulls_Pearson +=  Form("%3.3f", hist_pulls_pearson_toys->GetRMS());
  rms_pulls_Pearson += " #pm ";
  rms_pulls_Pearson += Form( "%3.3f", hist_pulls_pearson_toys->GetRMSError());

  pt->AddText(mean_pulls_Pearson);
  pt->AddText(rms_pulls_Pearson);
  pt->AddText(" ");


  TString dof = "D.o.F:  ";
  dof += numbins100160;
  dof += " - ";
  dof += number_of_free_background_parameters;

  TString numtoys = "# toys:  ";
  numtoys += number_of_toys;

  //pt->AddText(expected_rms);
  pt->AddText(dof);
  pt->AddText(numtoys);
  pt->Draw("same");

  //Gaussian distribution centered at zero, width set to the expected rms
  TF1 *gauss = new TF1("gauss","[0]*exp(-pow(x,2)/2*[1])/([1]*sqrt(2*3.14159265))",-5,5);
  gauss->FixParameter(0,hist_pulls_pearson_toys->GetSumOfWeights()*hist_pulls_pearson_toys->GetBinWidth(1)); 
  gauss->FixParameter(1,expectedrmsNumber);
  gauss->SetLineColor(2);
  gauss->Draw("same");

  TPaveText *gausstext = new TPaveText(0.19,0.77,0.34,0.84,"NDC");
  //gausstext->SetTextAlign(11);
  gausstext->SetTextSize(0.03);
  gausstext->SetTextColor(2);
  gausstext->SetFillColor(0);
  gausstext->SetBorderSize(0);
  TString gtxt = "with expected RMS ";
  gtxt += Form("%3.3f",expectedrmsNumber);
  gausstext->AddText("normal distribution");
  gausstext->AddText(gtxt);
  gausstext->Draw("same");

  // TString pullspdf = "CP";
  // pullspdf += category;
  // pullspdf += "_";
  // pullspdf += backgroundfunc;
  // pullspdf += "_backgroundAsimovInput_2011_pearsonToypulls_";
  // if(rebinInputHisto) {
  //   pullspdf += desiredGeVperBin;
  //   pullspdf += "GeVbins";
  // }
  // pullspdf.ReplaceAll(".","p"); //latex does not understand a dot in the middle of the name
  // pullspdf += ".pdf";
  // canv_pulls->Print(pullspdf);




  //-----------  Performance studies

  //It only makes sense to do performance studies if we fitted with a spurious signal 
  if(turnOffSpurious){
    cout << endl << "Fitting a spurious signal is turned off: no performance studies are evaluated." << endl;
  }
  else {

    // The expected significance for the signal at the maximum absolute spurious signal
    Double_t signalmassMaxAbsoluteSpurious=hexpectedSignal->GetBinContent(hexpectedSignal->FindBin(massMaxAbsoluteSpurious));
    Double_t spuriousSignalmassMaxAbsoluteSpurious =  hAamplitude->GetBinContent(hAamplitude->FindBin(massMaxAbsoluteSpurious));
    Double_t sigma0massMaxAbsoluteSpurious = hAamplitude->GetBinError(hAamplitude->FindBin(massMaxAbsoluteSpurious));
    Double_t significancemassMaxAbsoluteSpurious = signalmassMaxAbsoluteSpurious/sigma0massMaxAbsoluteSpurious; 
    Double_t significanceBiasmassMaxAbsoluteSpurious = spuriousSignalmassMaxAbsoluteSpurious / sigma0massMaxAbsoluteSpurious;
    Double_t percentBmassMaxAbsoluteSpurious = spuriousSignalmassMaxAbsoluteSpurious/sigma0massMaxAbsoluteSpurious*100;
    Double_t percentSmassMaxAbsoluteSpurious = spuriousSignalmassMaxAbsoluteSpurious/signalmassMaxAbsoluteSpurious*100;

    Double_t deltamassMaxAbsoluteSpurious = 0;
    Double_t sigma0primemassMaxAbsoluteSpurious = 0;
    Double_t significanceCoveredmassMaxAbsoluteSpurious = 0;  
    if(spuriousSignalmassMaxAbsoluteSpurious>=0) {
      deltamassMaxAbsoluteSpurious = sigma0massMaxAbsoluteSpurious*sqrt(pow(1.0+spuriousSignalmassMaxAbsoluteSpurious/signalmassMaxAbsoluteSpurious,2)-1.0);
      sigma0primemassMaxAbsoluteSpurious = sqrt(pow(sigma0massMaxAbsoluteSpurious,2)+pow(deltamassMaxAbsoluteSpurious,2));
      significanceCoveredmassMaxAbsoluteSpurious = signalmassMaxAbsoluteSpurious/sigma0primemassMaxAbsoluteSpurious;
    }
    //Leave exclusion out for now (it's more complicated)
    // else {
    // }


    // The expected significance for the signal at the maximum spurious signal
    Double_t signalmassMaxSpurious=hexpectedSignal->GetBinContent(hexpectedSignal->FindBin(massMaxSpurious));
    Double_t spuriousSignalmassMaxSpurious =  hAamplitude->GetBinContent(hAamplitude->FindBin(massMaxSpurious));
    Double_t sigma0massMaxSpurious = hAamplitude->GetBinError(hAamplitude->FindBin(massMaxSpurious));
    Double_t significancemassMaxSpurious = signalmassMaxSpurious/sigma0massMaxSpurious; // /hAamplitude->GetBinError(hAamplitude->FindBin(massMaxSpurious));
    Double_t significanceBiasmassMaxSpurious = spuriousSignalmassMaxSpurious / sigma0massMaxSpurious;
    Double_t percentBmassMaxSpurious = spuriousSignalmassMaxSpurious/sigma0massMaxSpurious*100;
    Double_t percentSmassMaxSpurious = spuriousSignalmassMaxSpurious/signalmassMaxSpurious*100;
    Double_t deltamassMaxSpurious = 0;
    Double_t sigma0primemassMaxSpurious = 0;
    Double_t significanceCoveredmassMaxSpurious = 0;
    if(spuriousSignalmassMaxSpurious>=0) {
      deltamassMaxSpurious = sigma0massMaxSpurious*sqrt(pow(1.0+spuriousSignalmassMaxSpurious/signalmassMaxSpurious,2)-1.0);
      sigma0primemassMaxSpurious = sqrt(pow(sigma0massMaxSpurious,2)+pow(deltamassMaxSpurious,2));
      significanceCoveredmassMaxSpurious = signalmassMaxSpurious/sigma0primemassMaxSpurious;
    }
    //Leave exclusion out for now (it's more complicated)
    // else {
    // }


    // The expected significance for the signal at 115 GeV
    Double_t signal115GeV            = hexpectedSignal->GetBinContent(hexpectedSignal->FindBin(115));
    Double_t spuriousSignal115GeV    = hAamplitude->GetBinContent(hAamplitude->FindBin(115));
    Double_t sigma0115GeV            = hAamplitude->GetBinError(hAamplitude->FindBin(115));
    Double_t chisquared115GeV        = hchisquared->GetBinContent(hAamplitude->FindBin(115));
    Double_t significance115GeV      = signal115GeV/sigma0115GeV; 
    Double_t significanceBias115GeV  = spuriousSignal115GeV / sigma0115GeV;
    Double_t percentB115GeV          = spuriousSignal115GeV/sigma0115GeV*100;
    Double_t percentS115GeV          = spuriousSignal115GeV/signal115GeV*100;
    //---   
    Double_t delta115GeV = 0;	     
    Double_t sigma0prime115GeV = 0;   
    Double_t significanceCovered115GeV = 0;
    if(spuriousSignal115GeV>=0) {
      delta115GeV =  sigma0115GeV*sqrt(pow(1.0+spuriousSignal115GeV/signal115GeV,2)-1.0);
      sigma0prime115GeV = sqrt(pow(sigma0115GeV,2)+pow(delta115GeV,2));
      significanceCovered115GeV = signal115GeV/sigma0prime115GeV;
    }
   

    // The expected significance for the signal at 125 GeV
    Double_t signal125GeV            = hexpectedSignal->GetBinContent(hexpectedSignal->FindBin(125));
    Double_t spuriousSignal125GeV    = hAamplitude->GetBinContent(hAamplitude->FindBin(125));
    Double_t sigma0125GeV            = hAamplitude->GetBinError(hAamplitude->FindBin(125));
    Double_t chisquared125GeV        = hchisquared->GetBinContent(hAamplitude->FindBin(125));
    Double_t significance125GeV      = signal125GeV/sigma0125GeV; 
    Double_t significanceBias125GeV  = spuriousSignal125GeV / sigma0125GeV;
    Double_t percentB125GeV          = spuriousSignal125GeV/sigma0125GeV*100;
    Double_t percentS125GeV          = spuriousSignal125GeV/signal125GeV*100;
    //---   
    Double_t delta125GeV = 0;	     
    Double_t sigma0prime125GeV = 0;   
    Double_t significanceCovered125GeV = 0;
    if(spuriousSignal125GeV>=0) {
      delta125GeV =  sigma0125GeV*sqrt(pow(1.0+spuriousSignal125GeV/signal125GeV,2)-1.0);
      sigma0prime125GeV = sqrt(pow(sigma0125GeV,2)+pow(delta125GeV,2));
      significanceCovered125GeV = signal125GeV/sigma0prime125GeV;
    }
   

    // The expected significance for the signal at 140 GeV
    Double_t signal140GeV            = hexpectedSignal->GetBinContent(hexpectedSignal->FindBin(140));
    Double_t spuriousSignal140GeV    = hAamplitude->GetBinContent(hAamplitude->FindBin(140));
    Double_t sigma0140GeV            = hAamplitude->GetBinError(hAamplitude->FindBin(140));
    Double_t chisquared140GeV        = hchisquared->GetBinContent(hAamplitude->FindBin(140));
    Double_t significance140GeV      = signal140GeV/sigma0140GeV; 
    Double_t significanceBias140GeV  = spuriousSignal140GeV / sigma0140GeV;
    Double_t percentB140GeV          = spuriousSignal140GeV/sigma0140GeV*100;
    Double_t percentS140GeV          = spuriousSignal140GeV/signal140GeV*100;
    //---   
    Double_t delta140GeV = 0;	     
    Double_t sigma0prime140GeV = 0;   
    Double_t significanceCovered140GeV = 0;
    if(spuriousSignal140GeV>=0) {
      delta140GeV =  sigma0140GeV*sqrt(pow(1.0+spuriousSignal140GeV/signal140GeV,2)-1.0);
      sigma0prime140GeV = sqrt(pow(sigma0140GeV,2)+pow(delta140GeV,2));
      significanceCovered140GeV = signal140GeV/sigma0prime140GeV;
    }
  
  
    cout << endl;
    cout << "Minimum  spurious  in 100-150  = " << minSpurious_110150 << endl;
    cout << "Maximum  spurious  in 100-150  = " << maxSpurious_110150 << endl;
    cout << "Maximum |spurious| in 100-150  = " << maxAbsoluteSpurious_110150 << endl;
    cout << "Maximum  spurious  in 120-130  = " << maxSpurious_120130 << endl;
  
    cout << endl;
    cout << "Mass with max  spurious   = " << massMaxSpurious << endl;
    cout << "Mass with min  spurious   = " << massMinSpurious << endl;
    cout << "Mass with no   spuriou    = " << massZeroSpurious << endl;
    cout << "Mass with max |spurious|  = " << massMaxAbsoluteSpurious << endl;

    cout << endl;
    cout << "+++++++++++++++++ Performance +++++++++++++++++++++++++++" << endl;
    cout << "Maximum significance bias (at mass " << mass_maxsignificanceBiasAsimov << "): " << maxsignificanceBiasAsimov << endl;
    cout << "(";
    cout  <<  (int)(hAamplitude->GetBinContent(hAamplitude->FindBin(mass_maxsignificanceBiasAsimov))/hexpectedSignal->GetBinContent(hexpectedSignal->FindBin(mass_maxsignificanceBiasAsimov))*100) << "% of signal, ";
    cout << (int)(maxsignificanceBiasAsimov*100) << "% of background uncertainty)" << endl;
      
    ofstream performancetempfile("performanceTemp.txt");

    Double_t chisqplacebo=-1.0;
    
    PrintPerformance(115,signal115GeV,spuriousSignal115GeV,percentS115GeV,percentB115GeV,sigma0115GeV,significance115GeV,significanceBias115GeV,delta115GeV,sigma0prime115GeV,significanceCovered115GeV,chisquared115GeV,1,performancetempfile);
  
    PrintPerformance(125,signal125GeV,spuriousSignal125GeV,percentS125GeV,percentB125GeV,sigma0125GeV,significance125GeV,significanceBias125GeV,delta125GeV,sigma0prime125GeV,significanceCovered125GeV,chisquared125GeV,1,performancetempfile);
  
    PrintPerformance(140,signal140GeV,spuriousSignal140GeV,percentS140GeV,percentB140GeV,sigma0140GeV,significance140GeV,significanceBias140GeV,delta140GeV,sigma0prime140GeV,significanceCovered140GeV,chisquared140GeV,1,performancetempfile);
  
    cout << endl << endl << "**************** Max spurious ************************" ;
    PrintPerformance(massMaxSpurious,signalmassMaxSpurious,spuriousSignalmassMaxSpurious,percentSmassMaxSpurious,percentBmassMaxSpurious,sigma0massMaxSpurious,significancemassMaxSpurious,significanceBiasmassMaxSpurious,deltamassMaxSpurious,sigma0primemassMaxSpurious,significanceCoveredmassMaxSpurious,chisqplacebo,1,performancetempfile);
  
    cout << endl << endl << "**************** Max |spurious| ************************" ;
    PrintPerformance(massMaxAbsoluteSpurious,signalmassMaxAbsoluteSpurious,spuriousSignalmassMaxAbsoluteSpurious,percentSmassMaxAbsoluteSpurious,percentBmassMaxAbsoluteSpurious,sigma0massMaxAbsoluteSpurious,significancemassMaxAbsoluteSpurious,significanceBiasmassMaxAbsoluteSpurious,deltamassMaxAbsoluteSpurious,sigma0primemassMaxAbsoluteSpurious,significanceCoveredmassMaxAbsoluteSpurious,chisqplacebo,1,performancetempfile);

    // Moved end of performance studies from here 17.09.2018 Alex and Simon
  
  cout << "-------------------------------------------------------------------------------------------------------------------" << endl << endl;

  
  // Print table of sigma_0's
  cout << "mH (GeV):   110  115  120  125  130  135  140  145  150" << endl;
  cout << "sigma_0 : ";
  cout << setprecision(3) << showpoint << setw(5)
       << hAamplitude->GetBinError(hAamplitude->FindBin(110)) << setw(5)
       << hAamplitude->GetBinError(hAamplitude->FindBin(115)) << setw(5)
       << hAamplitude->GetBinError(hAamplitude->FindBin(120)) << setw(5)
       << hAamplitude->GetBinError(hAamplitude->FindBin(125)) << setw(5)
       << hAamplitude->GetBinError(hAamplitude->FindBin(130)) << setw(5)
       << hAamplitude->GetBinError(hAamplitude->FindBin(135)) << setw(5)
       << hAamplitude->GetBinError(hAamplitude->FindBin(140)) << setw(5)
       << hAamplitude->GetBinError(hAamplitude->FindBin(145)) << setw(5)
       << hAamplitude->GetBinError(hAamplitude->FindBin(150))
       << endl;  
  cout << setprecision(6) << endl;
  
  
  // Print summary of fit errors and unacceptable error matrices
  cout << "Number of failed fits       : " << failedFitsMC << " (MC), " << failedFitsAsimov << " (Asimov), "
       << failedFitsToys << " (Toys)" << endl;
  cout << "Number of bad error matrices: " << errorMatrixProblemsMC 
       << " (MC), " << errorMatrixProblemsAsimov << " (Asimov), "
       << errorMatrixProblemsToys << " (Toys)" << endl << endl;
  
  
  if(backgroundAsimovInput) {
    cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
    cout << ">>>>>>>>>>>>>>> Experimental replacement of input histogram with background fit <<<<<<<<<<<<<<<<<" << endl;
    cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
  }


  
  
  if(writeResultToFiles) {
    
    //Print canvas to pdf file
    TString SaveCanvasFilename = "HSG1backgroundBiasStudies_";
    if(filename.Contains("diphox"))
      SaveCanvasFilename += "Diphox_C";
    else if(filename.Contains("sherpa"))
      SaveCanvasFilename += "Sherpa_C";  
    else if(filename.Contains("resbos"))
      SaveCanvasFilename += "Resbos_C";
    SaveCanvasFilename += category;
    SaveCanvasFilename += "_";
    SaveCanvasFilename += backgroundfunction->GetName();
    SaveCanvasFilename += Form("_%2.1ffb",luminosity);
    SaveCanvasFilename.ReplaceAll(".","p"); //latex does not understand a dot in the middle of the name

    myc->Print(SaveCanvasFilename+".pdf","pdf");
    canv_zoom->Print(SaveCanvasFilename+"_spuriousZoom.pdf","pdf");
    canv_bias->Print(SaveCanvasFilename+"_fullBias.pdf","pdf");
    canv_pulls->Print(SaveCanvasFilename+"_PearsonToypulls.pdf","pdf");
    

    
    //It only makes sense to do performance studies if we fitted with a spurious signal 
    //Print performance to txt file
    TString performancefileName = SaveCanvasFilename.ReplaceAll(".pdf","");
    performancefileName += "_performance.txt";
    ofstream performancefile(performancefileName);

    if(turnOffSpurious){
      performancefile << endl <<  "Fitting a spurious signal is turned off: no performance studies are evaluated." << endl;
      cout << endl <<  "Fitting a spurious signal is turned off: no performance studies are evaluated." << endl;
    }
    else {
      backgroundfitresult->Print(performancefileName);

      performancefile << endl;
      performancefile << "Minimum  spurious  in 100-150  = " << minSpurious_110150 << endl;
      performancefile << "Maximum  spurious  in 100-150  = " << maxSpurious_110150 << endl;
      performancefile << "Maximum |spurious| in 100-150  = " << maxAbsoluteSpurious_110150 << endl;
      performancefile << "Maximum  spurious  in 120-130  = " << maxSpurious_120130 << endl;
    
      performancefile << endl;
      performancefile << "Mass with max  spurious   = " << massMaxSpurious << endl;
      performancefile << "Mass with min  spurious   = " << massMinSpurious << endl;
      performancefile << "Mass with no   spuriou    = " << massZeroSpurious << endl;
      performancefile << "Mass with max |spurious|  = " << massMaxAbsoluteSpurious << endl;
    
      performancefile << endl;
      performancefile << "+++++++++++++++++ Performance +++++++++++++++++++++++++++" << endl;
      performancefile << "Maximum significance bias (at mass " << mass_maxsignificanceBiasAsimov << "): " << maxsignificanceBiasAsimov << endl;
      performancefile << "(";
      performancefile  <<  (int)(hAamplitude->GetBinContent(hAamplitude->FindBin(mass_maxsignificanceBiasAsimov))/hexpectedSignal->GetBinContent(hexpectedSignal->FindBin(mass_maxsignificanceBiasAsimov))*100) << "% of signal, ";
      performancefile << (int)(maxsignificanceBiasAsimov*100) << "% of background uncertainty)" << endl << endl;

      PrintPerformance(115,signal115GeV,spuriousSignal115GeV,percentS115GeV,percentB115GeV,sigma0115GeV,significance115GeV,significanceBias115GeV,delta115GeV,sigma0prime115GeV,significanceCovered115GeV,chisquared115GeV,0,performancefile);

      PrintPerformance(125,signal125GeV,spuriousSignal125GeV,percentS125GeV,percentB125GeV,sigma0125GeV,significance125GeV,significanceBias125GeV,delta125GeV,sigma0prime125GeV,significanceCovered125GeV,chisquared125GeV,0,performancefile);

      PrintPerformance(140,signal140GeV,spuriousSignal140GeV,percentS140GeV,percentB140GeV,sigma0140GeV,significance140GeV,significanceBias140GeV,delta140GeV,sigma0prime140GeV,significanceCovered140GeV,chisquared140GeV,0,performancefile);

      performancefile << endl << endl << "**************** Max spurious ************************" ;
      PrintPerformance(massMaxSpurious,signalmassMaxSpurious,spuriousSignalmassMaxSpurious,percentSmassMaxSpurious,percentBmassMaxSpurious,sigma0massMaxSpurious,significancemassMaxSpurious,significanceBiasmassMaxSpurious,deltamassMaxSpurious,sigma0primemassMaxSpurious,significanceCoveredmassMaxSpurious,chisqplacebo,0,performancefile);

      performancefile << endl << endl << "**************** Max |spurious| ************************" ;
      PrintPerformance(massMaxAbsoluteSpurious,signalmassMaxAbsoluteSpurious,spuriousSignalmassMaxAbsoluteSpurious,percentSmassMaxAbsoluteSpurious,percentBmassMaxAbsoluteSpurious,sigma0massMaxAbsoluteSpurious,significancemassMaxAbsoluteSpurious,significanceBiasmassMaxAbsoluteSpurious,deltamassMaxAbsoluteSpurious,sigma0primemassMaxAbsoluteSpurious,significanceCoveredmassMaxAbsoluteSpurious,chisqplacebo,0,performancefile);

    }

    performancefile << "-------------------------------------------------------------------------------------------------------------------" << endl << endl;

    
    performancefile << endl 
		    << "--- Options ---" << endl
		    << "verbose:                " << verbose << endl
		    << "dologlikelihood:        " << dologlikelihood << endl
		    << "backgroundAsimovInput:  " << backgroundAsimovInput << endl
		    << "turnOffSpurious:        " << turnOffSpurious << endl
		    << "minuit2:                " <<  minuit2 << endl
		    << "minos:                  " << minos << endl
		    << "improve:                " << improve << endl 
		    << "showfitbands:           " << showfitbands << endl
		    << "writeResultToFiles:     " << writeResultToFiles << endl
		    << "rebinInputHisto:        " << rebinInputHisto << endl
		    << endl;


    // Print summary of fit errors and unacceptable error matrices
    performancefile << "Number of failed fits       : " << failedFitsMC << " (MC), " << failedFitsAsimov << " (Asimov), "
		    << failedFitsToys << " (Toys)" << endl;
    performancefile << "Number of bad error matrices: " << errorMatrixProblemsMC 
		    << " (MC), " << errorMatrixProblemsAsimov << " (Asimov), "
		    << errorMatrixProblemsToys << " (Toys)" << endl << endl;


    // Print table of sigma_0's
    performancefile << "mH (GeV):   110  115  120  125  130  135  140  145  150" << endl;
    performancefile << "sigma_0 : ";
    performancefile << setprecision(3) << showpoint << setw(5)
		    << hAamplitude->GetBinError(hAamplitude->FindBin(110)) << setw(5)
		    << hAamplitude->GetBinError(hAamplitude->FindBin(115)) << setw(5)
		    << hAamplitude->GetBinError(hAamplitude->FindBin(120)) << setw(5)
		    << hAamplitude->GetBinError(hAamplitude->FindBin(125)) << setw(5)
		    << hAamplitude->GetBinError(hAamplitude->FindBin(130)) << setw(5)
		    << hAamplitude->GetBinError(hAamplitude->FindBin(135)) << setw(5)
		    << hAamplitude->GetBinError(hAamplitude->FindBin(140)) << setw(5)
		    << hAamplitude->GetBinError(hAamplitude->FindBin(145)) << setw(5)
		    << hAamplitude->GetBinError(hAamplitude->FindBin(150))
		    << endl;  

      
    performancefile.close();


    //Latex table file
    TString  latextablefileName = SaveCanvasFilename.ReplaceAll(".pdf","");
    latextablefileName += "_latexTable.txt";
    ofstream latextablefile(latextablefileName);
    latextablefile.precision(3);
    latextablefile << "CP" << category << " & " << spuriousSignalmassMaxAbsoluteSpurious
		   << " $\\left( " << massMaxAbsoluteSpurious << " \\right)$ & "
		   << percentSmassMaxAbsoluteSpurious  << " $\\left( "  << signalmassMaxAbsoluteSpurious << " \\right)$ & "
		   << percentBmassMaxAbsoluteSpurious  << " $\\left( "  << sigma0massMaxAbsoluteSpurious << " \\right)$ & "
		   << significancemassMaxAbsoluteSpurious << " & " << significanceBiasmassMaxAbsoluteSpurious << " & " ;
    if(luminosity==4.9){
      if((fabs(percentBmassMaxAbsoluteSpurious) <= 14) || (fabs(percentSmassMaxAbsoluteSpurious)<=10) ) latextablefile << "$\\checkmark$";
    }
    if(luminosity==10){
      if((fabs(percentBmassMaxAbsoluteSpurious) <= 20) || (fabs(percentSmassMaxAbsoluteSpurious)<=10) ) latextablefile << "$\\checkmark$";
    }
    //latextablefile << " \\\\ \\hline" << endl;
    latextablefile << " \\\\" << endl;
    latextablefile.close();


  }
  } // New end of performance studies 17.09.2018 Alex and Simon - note: have to fix indentation up to line 1792!


  //significanceRootfile->Close();


  // TCanvas *canv_p0=new TCanvas("canv_p0","");
  // canv_p0->cd();

  // hp0_expected_asimov->GetXaxis()->SetTitle("M#gamma#gamma [GeV]");
  // hp0_expected_asimov->GetYaxis()->SetTitleSize(0.04);
  // hp0_expected_asimov->GetYaxis()->SetTitle("expected p0 (asimov)   "+title);
  // hp0_expected_asimov->GetYaxis()->SetTitleOffset(1.3);
  // hp0_expected_asimov->SetLineStyle(kDashed);
  // hp0_expected_asimov->SetLineWidth(4);
  // hp0_expected_asimov->SetLineColor(kRed); 
  // hp0_expected_asimov->SetAxisRange(hp0_expected_asimov->GetBinCenter(startBinFit),hp0_expected_asimov->GetBinCenter(stopBinFit-1),"X");
  // hp0_expected_asimov->SetAxisRange(1.e-04,10,"Y");
  // hp0_expected_asimov->Draw("l");


  // //Draw 1, 2 and 3 sigma lines

  // float onesigma = Pvalue_uncapped(1);
  // TLine *line1sigma = new TLine(startMassFit,onesigma,stopMassFit,onesigma);
  // line1sigma->SetLineStyle(8);
  // line1sigma->SetLineColor(kBlue);
  // line1sigma->SetLineWidth(3);
  // line1sigma->Draw();

  // float twosigma = Pvalue_uncapped(2);
  // TLine *line2sigma = new TLine(startMassFit,twosigma,stopMassFit,twosigma);
  // line2sigma->SetLineStyle(8);
  // line2sigma->SetLineColor(kBlue);
  // line2sigma->SetLineWidth(3);
  // line2sigma->Draw();

  // float threesigma = Pvalue_uncapped(3);
  // TLine *line3sigma = new TLine(startMassFit,threesigma,stopMassFit,threesigma);
  // line3sigma->SetLineStyle(8);
  // line3sigma->SetLineColor(kBlue);
  // line3sigma->SetLineWidth(3);
  // line3sigma->Draw();

  // canv_p0->SetLogy();



  return(0);
}



//Not necessary to store the parameters, don't bother to remove them
  void FixParametersOfSpuriousModel(int number_of_background_parameters, TF1 *model, double binwidth, double alphaCB, double sigmaG, double meanG, double fractionCB, double sigmaCB, double meanCB) {
  
  //parameter 0 of the signalmodel is the parameter at position number_of_background_parameters: the amplitude which shall be fitted
  // model->FixParameter(number_of_background_parameters+1, model->GetParameter(number_of_background_parameters+1));
  // model->FixParameter(number_of_background_parameters+2, model->GetParameter(number_of_background_parameters+2));
  // model->FixParameter(number_of_background_parameters+3, model->GetParameter(number_of_background_parameters+3));
  // model->FixParameter(number_of_background_parameters+4, model->GetParameter(number_of_background_parameters+4));
  // model->FixParameter(number_of_background_parameters+5, model->GetParameter(number_of_background_parameters+5));
  // model->FixParameter(number_of_background_parameters+6, model->GetParameter(number_of_background_parameters+6));
  // model->FixParameter(number_of_background_parameters+7, model->GetParameter(number_of_background_parameters+7));
  // model->FixParameter(number_of_background_parameters+8, model->GetParameter(number_of_background_parameters+8));
  // model->FixParameter(number_of_background_parameters+9, model->GetParameter(number_of_background_parameters+9));
  model->FixParameter(number_of_background_parameters, model->GetParameter(number_of_background_parameters));
  model->FixParameter(number_of_background_parameters+1, model->GetParameter(number_of_background_parameters+1));
  model->FixParameter(number_of_background_parameters+2, model->GetParameter(number_of_background_parameters+2));
  model->FixParameter(number_of_background_parameters+3, model->GetParameter(number_of_background_parameters+3));
  model->FixParameter(number_of_background_parameters+4, model->GetParameter(number_of_background_parameters+4));
  model->FixParameter(number_of_background_parameters+5, model->GetParameter(number_of_background_parameters+5));
  model->FixParameter(number_of_background_parameters+6, model->GetParameter(number_of_background_parameters+6));
  model->FixParameter(number_of_background_parameters+7, model->GetParameter(number_of_background_parameters+7));
  model->FixParameter(number_of_background_parameters+8, model->GetParameter(number_of_background_parameters+8));
  model->FixParameter(number_of_background_parameters+9, model->GetParameter(number_of_background_parameters+9));
  
 }
 
 
void PrintPerformance(Double_t mass, Double_t &expectedSignal, Double_t &spuriousSignal, Double_t &percentS, Double_t &percentB, Double_t &sigma0, Double_t &significance, Double_t &significanceBias, Double_t &delta, Double_t &sigma0prime, Double_t & significanceCovered, Double_t &chisquared, bool printToScreen, ofstream &file) {

  if(printToScreen) {
     cout << endl;
     cout << "--------------- At " << mass << " GeV ----------------------------------------------------------------------------------------" << endl;
     cout << "chi-squared for fit to MC      : " << chisquared << endl;
     cout << "Expected signal rate           : " << expectedSignal << endl;
     cout << "Spurious signal               : " << spuriousSignal << " (" 
	  << (Int_t)percentS << "% of signal, " << (Int_t)percentB << "% of background uncertainty)" << endl;
     cout << "sigma_0                        : " << sigma0 << endl;
     cout << "Unbiased expected significance : " << significance << endl;
     cout << "Biased expected significance   : " << significance+significanceBias << endl;
     cout << "Significance bias              : " << significanceBias << endl;
     if(spuriousSignal>=0) {
       cout << endl << "         ===== Concerning positive bias and effect on discovery ===== " << endl;
       cout << "Additional uncertainty required to \"cover\" bias for median signal: " <<  delta
	    << " (total=" << sigma0prime <<  ")" << endl;
       cout << "Expected (conservative) signal significance with spurious signal \"covered\": " << significanceCovered << endl;
     }      
  }
  else {
    file << endl;
    file << "--------------- At " << mass << " GeV ----------------------------------------------------------------------------------------" << endl;
    file << "Expected signal rate           : " << expectedSignal << endl;
    file << "Spurious signal               : " << spuriousSignal << " (" 
	 << (Int_t)percentS << "% of signal, " << (Int_t)percentB << "% of background uncertainty)" << endl;
    file << "sigma_0                        : " << sigma0 << endl;
    file << "Unbiased expected significance : " << significance << endl;
    file << "Biased expected significance   : " << significance+significanceBias << endl;
    file << "Significance bias              : " << significanceBias << endl;
    if(spuriousSignal>=0) {
      file << endl << "         ===== Concerning positive bias and effect on discovery ===== " << endl;
      file << "Additional uncertainty required to \"cover\" bias for median signal: " <<  delta
	   << " (total=" << sigma0prime <<  ")" << endl;
      file << "Expected (conservative) signal significance with spurious signal \"covered\": " << significanceCovered << endl;
    }
  }
}





// float number_of_signal_events(TString &histname, Double_t luminosity) {

//   float number_of_signal_events = 0;


//   //Numbers are from MC11c, mH=120

//   Int_t category=999;
//   if(histname.Contains("1")) {
//     category = 1;
//   }
//   if(histname.Contains("2"))
//     category = 2;
//     // number_of_signal_events = 2.99;
//   if(histname.Contains("3"))
//     category = 3;
//     // number_of_signal_events = 20.1;
//   if(histname.Contains("4"))
//     category = 4;
//     // number_of_signal_events = 5.95;
//   if(histname.Contains("5"))
//     category = 5;
//     // number_of_signal_events = 6.13;
//   if(histname.Contains("6"))
//     category = 6;
//     // number_of_signal_events = 1.94;
//   if(histname.Contains("7"))
//     category = 7;
//     // number_of_signal_events = 19.4;
//   if(histname.Contains("8"))
//     category = 8;
//     // number_of_signal_events = 5.91;
//   if(histname.Contains("9"))
//     category = 9;
//     // number_of_signal_events = 10.1;

//   vector<float> *sigpars = GetSigPars(category,"PttEtaConvCat",120,"all","mc11c","commonPDF");
//   number_of_signal_events = sigpars->at(par_nSig);
//   cout << "Number of signal events in category " << category << " for 4.9/fb: " << number_of_signal_events << endl;

//   return number_of_signal_events*luminosity/4.9;

// }



// Void eraselabel(TPad *p,Double_t h) {
//    p->cd();
//    pe = new TPad("pe","pe",0,0,p->GetLeftMargin(),h);       
//    pe->Draw(); 
//    pe->SetFillColor(p->GetFillColor());  
//    pe->SetBorderMode(0);
// }


//For debugging:
// int main(){

//   HSG1BackgroundBiasStudy(100,160,"diphox_shape.root","C1", "exp", 4);

// }


