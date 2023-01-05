public class Ex4_1 {
    public static void main (String[] args){
        // 1. 
        int x = 15;
        if(x>10 && x<20){
            System.out.println("true");
        } else {System.out.println("false");}

        // 2.
        char ch = ' ';
        if(!(ch==' ')){
        // 대입: =, 동일 여부: ==
            System.out.println("true");
        } else {System.out.println("false");}

        // 3.
        char ch2 = 'x';
        if(ch2 == 'x' || ch2 == 'X') {
            System.out.println("true");
        } else {System.out.println("false");}

        // 4.
        char ch3 = '3';
        if(ch3>='0' && ch3<='9'){
            System.out.println("true");
        } else {System.out.println("false");}

        // 5.
        char ch4 = 'a';
        if(ch4>=97 && ch4<=128) {
            System.out.println("ture");
        } else {System.out.println("false");}

        // 6.
        int year = 2023;
        if(year%400==0 || (year%4==0 && year%100!=0)){
            System.out.println("true");
        } else {System.out.println("false");}

        // 7.
        boolean powerOn = true;
        if(powerOn == false){
            System.out.println("true");
        } else {System.out.println("false");}

        // 8.
        String str = "yes";
        if(str == "yes"){
            System.out.println("true");
        } else {System.out.println("false");}

    }
}
