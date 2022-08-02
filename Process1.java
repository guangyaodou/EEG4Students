/**
Process1 this is a java program for pulling out just the 20 eeg bands that we care about from the MuseBand EEG data
 **/

/*
We are processing lines like this
1477582509.410340, /muse/elements/alpha_relative, 0.0476856231689, 0.347870320082, 0.0865249782801, 0.0705577433109
1477582509.410397, /muse/elements/beta_relative, 0.0532405897975, 0.241317227483, 0.0607733689249, 0.0912269204855
1477582509.410483, /muse/elements/delta_relative, 0.473148316145, 0.101762481034, 0.527960002422, 0.673011481762
1477582509.410519, /muse/elements/gamma_relative, 0.0333921946585, 0.160039246082, 0.0651345327497, 0.0189646780491
1477582509.410583, /muse/elements/theta_relative, 0.392533272505, 0.149010747671, 0.259607106447, 0.146239206195
http://developer.choosemuse.com/tools/windows-tools/available-data-muse-direct
change to
/elements/delta_absolute
/elements/theta_absolute
/elements/alpha_absolute
/elements/beta_absolute
/elements/gamma_absolute
*/


import java.util.Scanner;
import java.io.File;

public class Process1{

    public static void main(String[] args){
		Scanner scanner = new Scanner(System.in);
		int counter=0;
		while (scanner.hasNext()){
		    String line = scanner.nextLine();
		    int pos = line.indexOf("_absolute");
		    String result = "";
		    if (pos > -1) {
			counter++;
			for(int i=0; i<5;i++){
			    result += processLine(line,i==0);
			    line = scanner.nextLine();
			}
			if (result.indexOf("nan")==-1)
			    System.out.println(result);
		    }
		}
    }


    public static String processLine(String line,boolean first){
		Scanner scan = new Scanner(line).useDelimiter(",");
		String result="";
		if (first) result=scan.next()+" ";
		else scan.next();
		scan.next();
		result += scan.next()+" ";
		result += scan.next()+" ";
		result += scan.next()+" ";
		result += scan.next()+" ";
		return result;
    }


}
