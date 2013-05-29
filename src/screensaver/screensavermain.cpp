#include <winsock2.h>
#include <windows.h>
#include <scrnsave.h>
#include <commctrl.h>

#include <direct.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <sstream>

#include "../rpcminer/rpcminerclient.h"
#include "../rpcminer/hex.h"
#include "resource.h"
#include "registry/Registry.h"

#ifdef UNICODE
#pragma comment(lib, "scrnsavw.lib")
#else
#pragma comment(lib, "scrnsave.lib")
#endif
#pragma comment(lib, "comctl32.lib")


#pragma region globals
HWND winhwnd ;
HBITMAP backBMP ;
HDC backDC ;
int backBufferCX, backBufferCY ;

#define TIMER 1
RPCMinerClient *rpcclient=0;
bool threadrunning=false;
HANDLE hthread;
HICON bcicon=0;
std::vector<std::pair<int,int> > iconpos;

int64 hashrate=0;
int64 lastgothashrate=0;
std::map<std::string,std::string> mapArgs;
std::map<std::string,std::vector<std::string> > mapMultiArgs;
#pragma endregion

const bool Convert(LPCWSTR input, std::string &output)
{
	if(input==NULL)
	{
		return true;
	}
	int inputlen=lstrlenW(input);
	if(inputlen==0)
	{
		return true;
	}
	int len=WideCharToMultiByte(CP_UTF8,0,input,inputlen,NULL,0,NULL,NULL);
	if(len==0)
	{
		return false;
	}
	char *out=new char[len+1];
	len=WideCharToMultiByte(CP_UTF8,0,input,inputlen,out,len,NULL,NULL);
	if(len==0)
	{
		delete [] out;
		return false;
	}
	out[len]='\0';
	output=out;

	delete [] out;

	return true;
}

const bool Convert(const std::string &input, std::wstring &output)
{
	int len=MultiByteToWideChar(CP_UTF8,0,input.c_str(),input.size(),NULL,0);
	if(len==0)
	{
		return true;
	}
	LPWSTR str=new WCHAR[len+1];
	len=MultiByteToWideChar(CP_UTF8,0,input.c_str(),input.size(),str,len);
	if(len==0)
	{
		delete [] str;
		return false;
	}
	str[len]='\0';
	output=str;

	delete [] str;

	return true;
}

void RunClientThread(void *param)
{
	TCHAR path[MAX_PATH]={0};
	GetModuleFileName(NULL,path,MAX_PATH);
	std::string outpath("");
	Convert(path,outpath);
	std::string::size_type pos=outpath.rfind("/");
	if(pos!=std::string::npos)
	{
		outpath.erase(pos);
	}
	else
	{
		pos=outpath.rfind("\\");
		if(pos!=std::string::npos)
		{
			outpath.erase(pos);
		}
	}

	_chdir(outpath.c_str());

	threadrunning=true;
	if(rpcclient)
	{
		rpcclient->Stop();
		while(rpcclient->Running())
		{
			Sleep(10);
		}
		delete rpcclient;
		rpcclient=0;
	}
	rpcclient=new RPCMinerClient;
	if(rpcclient)
	{
		// get reg values and run client
		CRegistry reg;
		int workrefreshms=0;
		std::string url("");
		std::string user("");
		std::string password("");
		int threadcount=1;

		rpcclient->SetHashRateRefresh(1000000);

		reg.Open(_T("Software\\Bitcoin Screensaver"),HKEY_CURRENT_USER);

		if(reg[_T("workrefreshms")].Exists()==true && reg[_T("workrefreshms")].IsDWORD()==true)
		{
			rpcclient->SetWorkRefreshMS((DWORD)reg[_T("workrefreshms")]);
		}

		if(reg[_T("url")].Exists()==true && reg[_T("url")].IsString()==true)
		{
			Convert(reg[_T("url")],url);
		}

		if(reg[_T("username")].Exists()==true && reg[_T("username")].IsString()==true)
		{
			Convert(reg[_T("username")],user);
		}

		if(reg[_T("password")].Exists()==true && reg[_T("password")].IsString()==true)
		{
			Convert(reg[_T("password")],password);
		}

		if(reg[_T("threadscpu")].Exists()==true && reg[_T("threadscpu")].IsDWORD()==true)
		{
			threadcount=(DWORD)reg[_T("threadscpu")];
		}
		else
		{
#if !defined(_BITCOIN_MINER_CUDA_) && !defined(_BITCOIN_MINER_OPENCL_)
			threadcount=boost::thread::hardware_concurrency();
#endif
		}

#if defined(_BITCOIN_MINER_CUDA_) || defined(_BITCOIN_MINER_OPENCL_)
		if(reg[_T("gpu")].Exists()==true && reg[_T("gpu")].IsDWORD()==true)
		{
			std::ostringstream ostr;
			ostr << (DWORD)reg[_T("gpu")];
			mapArgs["-gpu"]=ostr.str();
		}
		if(reg[_T("platform")].Exists()==true && reg[_T("platform")].IsDWORD()==true)
		{
			std::ostringstream ostr;
			ostr << (DWORD)reg[_T("platform")];
			mapArgs["-platform"]=ostr.str();
		}
		if(reg[_T("aggression")].Exists()==true && reg[_T("aggression")].IsDWORD()==true)
		{
			std::ostringstream ostr;
			ostr << (DWORD)reg[_T("aggression")];
			mapArgs["-aggression"]=ostr.str();
		}
		if(reg[_T("grid")].Exists()==true && reg[_T("grid")].IsDWORD()==true)
		{
			std::ostringstream ostr;
			ostr << (DWORD)reg[_T("grid")];
			mapArgs["-gpugrid"]=ostr.str();
		}
		if(reg[_T("threadsgpu")].Exists()==true && reg[_T("threadsgpu")].IsDWORD()==true)
		{
			std::ostringstream ostr;
			ostr << (DWORD)reg[_T("threadsgpu")];
			mapArgs["-gputhreads"]=ostr.str();
		}
#endif

		rpcclient->Run(url,user,password,threadcount);
		delete rpcclient;
		rpcclient=0;
	}
	threadrunning=false;
}

LRESULT WINAPI ScreenSaverProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	int tempcount=0;
  switch(message)
  {
  case WM_CREATE:
    {
      // Create a compatible bitmap with the width, height
      // of the desktop.
      winhwnd = hwnd ; // save.

      backBufferCX = GetSystemMetrics( SM_CXVIRTUALSCREEN ) ;  // SM_CXSCREEN is just the primary.  this is BOTH.
      backBufferCY = GetSystemMetrics( SM_CYVIRTUALSCREEN ) ;

      //HDC desktopHdc = GetDC( NULL ) ; // give me the hdc of the desktop.
      HDC winhdc = GetDC( winhwnd ) ; // give me the hdc of this app.
	  if(fChildPreview)
	  {
		RECT rec;
		GetClientRect(hwnd,&rec);
		backBufferCX=(rec.right-rec.left)+1;
		backBufferCY=(rec.bottom-rec.top)+1;
	  }

	backBMP = (HBITMAP)CreateCompatibleBitmap( winhdc, backBufferCX, backBufferCY ) ;

	iconpos.clear();
	iconpos.resize(500,std::pair<int,int>(-1,-1));
	srand(time(0));

      // now, you need to associate a dc with this
      // bitmap.  the DC holds all the info about brushes, etc.
      backDC = CreateCompatibleDC( winhdc ) ;
      ReleaseDC( winhwnd, winhdc ) ;

      // select it in.  here we have to associate
      // the DC we made (compatible with the window's hdc)
      // with the bitmap we made (compatible with the window's bitmap)
      SelectObject( backDC, backBMP ) ;

      //char buf[300] ;
      //sprintf( buf, "desktop width:  %d     height:  %d", backBufferCX, backBufferCY ) ;
      //TextOutA( backDC, 20, 20, buf, strlen( buf ) );

      // Timer for animation
      SetTimer( winhwnd, TIMER, 24, NULL );

	  if(!fChildPreview)
	  {
		if(bcicon==0)
		{
			bcicon=LoadIcon(GetModuleHandle(NULL),MAKEINTRESOURCE(IDI_BITCOINICON));
		}
		hthread=0;
		hthread=CreateThread(RunClientThread,NULL,false);
	  }
    }
    break;
  case WM_DESTROY:
	  if(rpcclient)
	  {
		rpcclient->Stop();
	  }
	  while(threadrunning==true && tempcount++<1000)
	  {
		Sleep(10);
	  }
	  if(hthread!=0)
	  {
		TerminateThread(hthread,0);
	  }
	  if(bcicon)
	  {
		DestroyIcon(bcicon);
	  }
    PostQuitMessage(0);
    break;
  case WM_PAINT:
    // blit the back buffer onto the screen
    {
      PAINTSTRUCT ps ;

      HDC hdc = BeginPaint( hwnd, &ps );
      BitBlt( hdc, 0, 0, backBufferCX, backBufferCY, backDC, 0, 0, SRCCOPY ) ;

      EndPaint( hwnd, &ps ) ;
    }
    break;

  case WM_TIMER:

	  if(!fChildPreview)
	  {

    // draw an extra point in a random spot.
    //SetPixel( backDC, rand()%backBufferCX, rand()%backBufferCY, RGB( 255,0,0 ) ) ;

      // clear by filling out the back buffer with black rectangle
      HBRUSH oldBrush = (HBRUSH)SelectObject( backDC, GetStockObject( BLACK_BRUSH ) ) ;
      Rectangle( backDC, 0, 0, backBufferCX, backBufferCY );
      SelectObject( backDC, oldBrush ) ; // put the old brush back

	if(bcicon)
	{
		for(std::vector<std::pair<int,int> >::const_iterator i=iconpos.begin(); i!=iconpos.end(); i++)
		{
			if((*i).first>-1 && (*i).second>-1)
			{
				DrawIcon(backDC,(*i).first,(*i).second,bcicon);
			}
		}
		int r=rand()%iconpos.size();
		iconpos[r].first=rand()%backBufferCX;
		iconpos[r].second=rand()%backBufferCY;
	}
	else
	{
		TextOut( backDC, rand()%backBufferCX, rand()%backBufferCY, _T("BTC"), 3 );
	}

	if(rpcclient)
	{

		std::ostringstream threadcountostr;
		threadcountostr << "CPU Threads : " << rpcclient->GetThreadCount();
		{
			std::string threadcountstr(threadcountostr.str());
			TCHAR *threadcounttc=new TCHAR[threadcountstr.size()+1];
			threadcounttc[threadcountstr.size()]=0;
			std::copy(threadcountstr.begin(),threadcountstr.end(),threadcounttc);
			TextOut(backDC,30,30,threadcounttc,threadcountstr.size());
			delete [] threadcounttc;
		}

		uint256 lt=rpcclient->GetLastTarget();
		std::string targetstr=Hex::Reverse(lt.GetHex());
		if(rpcclient->HasWork()==true)
		{
			targetstr="Target : "+targetstr;
		}
		else
		{
			targetstr="No work available for hashing";
		}
		if(targetstr.size()>0)
		{
			TCHAR *targettc=new TCHAR[targetstr.size()+1];
			targettc[targetstr.size()]=0;
			std::copy(targetstr.begin(),targetstr.end(),targettc);
			TextOut(backDC,30,50,targettc,targetstr.size());
			delete [] targettc;
		}

		std::ostringstream hashrateostr;
		hashrateostr << "Hash Rate : " << hashrate << " khash/s";
		{
			std::string hashratestr(hashrateostr.str());
			TCHAR *hashratetc=new TCHAR[hashratestr.size()+1];
			hashratetc[hashratestr.size()]=0;
			std::copy(hashratestr.begin(),hashratestr.end(),hashratetc);
			TextOut(backDC,30,70,hashratetc,hashratestr.size());
			delete [] hashratetc;
		}

		int64 msnow=GetTimeMillis();
		if(lastgothashrate+10000<msnow)
		{
			hashrate=rpcclient->GetHashRate(true);
			lastgothashrate=msnow;
		}

		std::ostringstream blocksostr;
		blocksostr << "Blocks Found : " << rpcclient->GetBlocksFound();
		{
			std::string blockstr(blocksostr.str());
			TCHAR *blocktc=new TCHAR[blockstr.size()+1];
			blocktc[blockstr.size()]=0;
			std::copy(blockstr.begin(),blockstr.end(),blocktc);
			TextOut(backDC,30,90,blocktc,blockstr.size());
			delete [] blocktc;
		}
	}

	/*
    if( totalPoints > 300 )
    {
      // clear by filling out the back buffer with black rectangle
      HBRUSH oldBrush = (HBRUSH)SelectObject( backDC, GetStockObject( BLACK_BRUSH ) ) ;
      Rectangle( backDC, 0, 0, backBufferCX, backBufferCY );
      SelectObject( backDC, oldBrush ) ; // put the old brush back
      // Not keeping the system's BLACK_BRUSH

      totalPoints = 0 ;
    }
	*/
	  }
	  else
	  {
		  if(bcicon==0)
		  {
			bcicon=LoadIcon(GetModuleHandle(NULL),MAKEINTRESOURCE(IDI_BITCOINICON));
		  }
		  if(bcicon)
		  {
			DrawIcon(backDC,(backBufferCX/2)-16,(backBufferCY/2)-16,bcicon);
		  }
	  }

    RECT r ;
    GetClientRect( winhwnd, &r );
    InvalidateRect( hwnd, &r, false ) ;

    break;
  default:
    return DefScreenSaverProc(hwnd, message, wParam, lParam);
  }
  return 0;
}

void loaddialogvalues(HWND hDlg)
{
	CRegistry reg;

	reg.Open(_T("Software\\Bitcoin Screensaver"),HKEY_CURRENT_USER);

	if(reg[_T("workrefreshms")].Exists()==true && reg[_T("workrefreshms")].IsDWORD()==true)
	{
		SetDlgItemInt(hDlg,IDC_WORKREFRESHMS,(DWORD)reg[_T("workrefreshms")],TRUE);
	}
	else
	{
		SetDlgItemInt(hDlg,IDC_WORKREFRESHMS,4000,TRUE);
	}

	if(reg[_T("url")].Exists()==true && reg[_T("url")].IsString()==true)
	{
		SetDlgItemText(hDlg,IDC_URL,reg[_T("url")]);
	}
	else
	{
		SetDlgItemText(hDlg,IDC_URL,_T(""));
	}

	if(reg[_T("username")].Exists()==true && reg[_T("username")].IsString()==true)
	{
		SetDlgItemText(hDlg,IDC_USERNAME,reg[_T("username")]);
	}
	else
	{
		SetDlgItemText(hDlg,IDC_USERNAME,_T(""));
	}

	if(reg[_T("password")].Exists()==true && reg[_T("password")].IsString()==true)
	{
		SetDlgItemText(hDlg,IDC_PASSWORD,reg[_T("password")]);
	}
	else
	{
		SetDlgItemText(hDlg,IDC_PASSWORD,_T(""));
	}

	if(reg[_T("statsurl")].Exists()==true && reg[_T("statsurl")].IsString()==true)
	{
		SetDlgItemText(hDlg,IDC_STATSURL,reg[_T("statsurl")]);
	}
	else
	{
		SetDlgItemText(hDlg,IDC_STATSURL,_T(""));
	}

	if(reg[_T("threadscpu")].Exists()==true && reg[_T("threadscpu")].IsDWORD()==true)
	{
		SetDlgItemInt(hDlg,IDC_THREADSCPU,(DWORD)reg[_T("threadscpu")],TRUE);
	}
	else
	{
		SetDlgItemText(hDlg,IDC_THREADSCPU,_T(""));
	}

#if !defined(_BITCOIN_MINER_CUDA_) && !defined(_BITCOIN_MINER_OPENCL_)
	EnableWindow(GetDlgItem(hDlg,IDC_GPU),FALSE);
	EnableWindow(GetDlgItem(hDlg,IDC_PLATFORM),FALSE);
	EnableWindow(GetDlgItem(hDlg,IDC_AGGRESSION),FALSE);
	EnableWindow(GetDlgItem(hDlg,IDC_GRID),FALSE);
	EnableWindow(GetDlgItem(hDlg,IDC_THREADSGPU),FALSE);
#else
	if(reg[_T("gpu")].Exists()==true && reg[_T("gpu")].IsDWORD()==true)
	{
		SetDlgItemInt(hDlg,IDC_GPU,(DWORD)reg[_T("gpu")],TRUE);
	}
	else
	{
		SetDlgItemText(hDlg,IDC_GPU,_T(""));
	}

#ifndef _BITCOIN_MINER_OPENCL_
	EnableWindow(GetDlgItem(hDlg,IDC_PLATFORM),FALSE);
#else
	if(reg[_T("platform")].Exists()==true && reg[_T("platform")].IsDWORD()==true)
	{
		SetDlgItemInt(hDlg,IDC_PLATFORM,(DWORD)reg[_T("platform")],TRUE);
	}
	else
	{
		SetDlgItemText(hDlg,IDC_PLATFORM,_T(""));
	}
#endif

	if(reg[_T("aggression")].Exists()==true && reg[_T("aggression")].IsDWORD()==true)
	{
		SetDlgItemInt(hDlg,IDC_AGGRESSION,(DWORD)reg[_T("aggression")],TRUE);
	}
	else
	{
		SetDlgItemText(hDlg,IDC_AGGRESSION,_T(""));
	}

	if(reg[_T("grid")].Exists()==true && reg[_T("grid")].IsDWORD()==true)
	{
		SetDlgItemInt(hDlg,IDC_GRID,(DWORD)reg[_T("grid")],TRUE);
	}
	else
	{
		SetDlgItemText(hDlg,IDC_GRID,_T(""));
	}

	if(reg[_T("threadsgpu")].Exists()==true && reg[_T("threadsgpu")].IsDWORD()==true)
	{
		SetDlgItemInt(hDlg,IDC_THREADSGPU,(DWORD)reg[_T("threadsgpu")],TRUE);
	}
	else
	{
		SetDlgItemText(hDlg,IDC_THREADSGPU,_T(""));
	}

#endif

}

void savedialogvalues(HWND hDlg)
{
	BOOL tempbool;
	UINT tempuint;
	TCHAR temptchar[256]={0};
	CRegistry reg;

	reg.Open(_T("Software\\Bitcoin Screensaver"),HKEY_CURRENT_USER);

	tempuint=GetDlgItemInt(hDlg,IDC_WORKREFRESHMS,&tempbool,TRUE);
	if(tempbool==TRUE)
	{
		reg[_T("workrefreshms")]=(DWORD)(tempuint);
	}
	else
	{
		reg[_T("workrefreshms")].Delete();
	}

	GetDlgItemText(hDlg,IDC_URL,temptchar,256);
	reg[_T("url")]=temptchar;

	GetDlgItemText(hDlg,IDC_USERNAME,temptchar,256);
	reg[_T("username")]=temptchar;

	GetDlgItemText(hDlg,IDC_PASSWORD,temptchar,256);
	reg[_T("password")]=temptchar;

	GetDlgItemText(hDlg,IDC_STATSURL,temptchar,256);
	reg[_T("statsurl")]=temptchar;

	tempuint=GetDlgItemInt(hDlg,IDC_THREADSCPU,&tempbool,TRUE);
	if(tempbool==TRUE)
	{
		reg[_T("threadscpu")]=(DWORD)(tempuint);
	}
	else
	{
		reg[_T("threadscpu")].Delete();
	}

#if defined(_BITCOIN_MINER_CUDA_) || defined(_BITCOIN_MINER_OPENCL_)
	tempuint=GetDlgItemInt(hDlg,IDC_GPU,&tempbool,TRUE);
	if(tempbool==TRUE)
	{
		reg[_T("gpu")]=(DWORD)(tempuint);
	}
	else
	{
		reg[_T("gpu")].Delete();
	}

#ifdef _BITCOIN_MINER_OPENCL_
	tempuint=GetDlgItemInt(hDlg,IDC_PLATFORM,&tempbool,TRUE);
	if(tempbool==TRUE)
	{
		reg[_T("platform")]=(DWORD)(tempuint);
	}
	else
	{
		reg[_T("platform")].Delete();
	}
#endif

	tempuint=GetDlgItemInt(hDlg,IDC_AGGRESSION,&tempbool,TRUE);
	if(tempbool==TRUE)
	{
		reg[_T("aggression")]=(DWORD)(tempuint);
	}
	else
	{
		reg[_T("aggression")].Delete();
	}

	tempuint=GetDlgItemInt(hDlg,IDC_GRID,&tempbool,TRUE);
	if(tempbool==TRUE)
	{
		reg[_T("grid")]=(DWORD)(tempuint);
	}
	else
	{
		reg[_T("grid")].Delete();
	}

	tempuint=GetDlgItemInt(hDlg,IDC_THREADSGPU,&tempbool,TRUE);
	if(tempbool==TRUE)
	{
		reg[_T("threadsgpu")]=(DWORD)(tempuint);
	}
	else
	{
		reg[_T("threadsgpu")].Delete();
	}

#endif
}

BOOL WINAPI ScreenSaverConfigureDialog(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    //static HWND hSpeed;   // handle to speed scroll bar
    static HWND hOK;      // handle to OK push button

    switch(message) 
    { 
        case WM_INITDIALOG: 
 
            // Retrieve the application name from the .rc file.  
            //LoadString(hMainInstance, idsAppName, szAppName, 
            //           80 * sizeof(TCHAR)); 
 
            // Retrieve the .ini (or registry) file name. 
            //LoadString(hMainInstance, idsIniFile, szIniFile, 
            //           MAXFILELEN * sizeof(TCHAR)); 
 
            // TODO: Add error checking to verify LoadString success
            //       for both calls.
			
            // Retrieve any redraw speed data from the registry. 
            //lSpeed = GetPrivateProfileInt(szAppName, szRedrawSpeed, 
            //                              DEFVEL, szIniFile); 
 
            // If the initialization file does not contain an entry 
            // for this screen saver, use the default value. 
            //if(lSpeed > MAXVEL || lSpeed < MINVEL) 
            //    lSpeed = DEFVEL; 
 
            // Initialize the redraw speed scroll bar control.
            //hSpeed = GetDlgItem(hDlg, ID_SPEED); 
            //SetScrollRange(hSpeed, SB_CTL, MINVEL, MAXVEL, FALSE); 
            //SetScrollPos(hSpeed, SB_CTL, lSpeed, TRUE); 
 
            // Retrieve a handle to the OK push button control.  
            //hOK = GetDlgItem(hDlg, IDOK); 

			loaddialogvalues(hDlg);
 
            return TRUE; 
 
        case WM_HSCROLL: 

            // Process scroll bar input, adjusting the lSpeed 
            // value as appropriate. 
            switch (LOWORD(wParam)) 
                { 
                    case SB_PAGEUP: 
                        //--lSpeed; 
                    break; 
 
                    case SB_LINEUP: 
                        //--lSpeed; 
                    break; 
 
                    case SB_PAGEDOWN: 
                        //++lSpeed; 
                    break; 
 
                    case SB_LINEDOWN: 
                        //++lSpeed; 
                    break; 
 
                    case SB_THUMBPOSITION: 
                        //lSpeed = HIWORD(wParam); 
                    break; 
 
                    case SB_BOTTOM: 
                        //lSpeed = MINVEL; 
                    break; 
 
                    case SB_TOP: 
                        //lSpeed = MAXVEL; 
                    break; 
 
                    case SB_THUMBTRACK: 
                    case SB_ENDSCROLL: 
                        return TRUE; 
                    break; 
                } 

                //if ((int) lSpeed <= MINVEL) 
                //    lSpeed = MINVEL; 
                //if ((int) lSpeed >= MAXVEL) 
                //    lSpeed = MAXVEL; 
 
                //SetScrollPos((HWND) lParam, SB_CTL, lSpeed, TRUE); 
            break; 
 
        case WM_COMMAND: 
            switch(LOWORD(wParam)) 
            { 
                case IDOK: 
 
                    // Write the current redraw speed variable to
                    // the .ini file. 
                    //hr = StringCchPrintf(szTemp, 20, "%ld", lSpeed);
                    //if (SUCCEEDED(hr))
                    //    WritePrivateProfileString(szAppName, szRedrawSpeed, 
                    //                              szTemp, szIniFile); 

					savedialogvalues(hDlg);

					EndDialog(hDlg, LOWORD(wParam) == IDOK);
 
                case IDCANCEL: 
                    EndDialog(hDlg, LOWORD(wParam) == IDOK); 

                return TRUE; 
            } 
    } 
    return FALSE; 

}

BOOL WINAPI RegisterDialogClasses(HANDLE hInst)
{
  return TRUE;
}
